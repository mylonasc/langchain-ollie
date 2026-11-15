"""Optional real OTLP/protobuf transport to an isolated loopback-only receiver."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.otel


def test_real_agent_otlp_http_to_local_collector(tmp_path):
    code = """
import json, os, runpy, socket, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from langchain_ollie import RecordingSession, SpanKind, SpanLink
from langchain_ollie.exporters.otel import OpenTelemetryProcessor
os.environ['LANGSMITH_TRACING']='false'
os.environ['LANGCHAIN_TRACING_V2']='false'
for key in list(os.environ):
    if key.endswith('API_KEY'):
        del os.environ[key]
received=[]
class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == '/v1/traces'
        size=int(self.headers['Content-Length'])
        assert size < 5_000_000
        received.append(ExportTraceServiceRequest.FromString(self.rfile.read(size)))
        self.send_response(200)
        self.send_header('Content-Type','application/x-protobuf')
        self.send_header('Content-Length','0')
        self.end_headers()
    def log_message(self,*args):
        pass
server=HTTPServer(('127.0.0.1',0),Receiver)
port=server.server_address[1]
original=socket.socket.connect
def loopback_only(self,address):
    assert address == ('127.0.0.1',port), address
    return original(self,address)
socket.socket.connect=loopback_only
def forbidden(*args):
    raise AssertionError('unexpected connect_ex')
socket.socket.connect_ex=forbidden
worker=threading.Thread(target=server.serve_forever,daemon=True)
worker.start()
try:
    exporter=OTLPSpanExporter(endpoint=f'http://127.0.0.1:{port}/v1/traces',timeout=2)
    bridge=OpenTelemetryProcessor(exporter)
    examples=Path(sys.argv[1])
    sys.path.insert(0,str(examples))
    app=runpy.run_path(str(examples/'local_agent.py'))['build_agent']()
    with RecordingSession(processors=[bridge],strict=True,service_name='collector-test') as session:
        result=app.invoke({'messages':[('user','OTLP-PRIVATE-CANARY')]},
                          config={'callbacks':[session.handler]})
        previous=next(s for s in session.snapshot()[0].spans if s.parent_span_id is None)
        session.recorder.start_span('linked','linked',SpanKind.CUSTOM,
            links=[SpanLink(previous.trace_id,previous.span_id,'follows_from')])
        session.recorder.end_span('linked')
    assert result['messages'][-1].content == 'Result: 5'
    resources=[r for request in received for r in request.resource_spans]
    spans=[s for r in resources for scope in r.scope_spans for s in scope.spans]
    originals=[s for b in session.snapshot() for s in b.spans]
    assert {s.span_id.hex() for s in spans} == {str(s.span_id) for s in originals}
    assert len(spans)==len(originals)
    for r in resources:
        attrs={a.key:a.value.string_value for a in r.resource.attributes}
        assert attrs['service.name']=='collector-test'
    root=next(s for s in spans if s.name == 'Graph: LocalAgent')
    assert not root.parent_span_id
    assert any(s.parent_span_id == root.span_id for s in spans)
    linked=next(s for s in spans if s.name == 'linked')
    assert linked.links[0].span_id == root.span_id
    assert linked.links[0].trace_id == root.trace_id
    assert not any(b'OTLP-PRIVATE-CANARY' in r.SerializeToString() for r in received)
    session.export(sys.argv[2])
    assert not json.loads(Path(sys.argv[2]).read_text())['otherData']['incomplete']
    assert bridge.shutdown(timeout=2)
finally:
    server.shutdown()
    server.server_close()
    worker.join(5)
    assert not worker.is_alive()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).parents[1] / "examples"),
            str(tmp_path / "trace.json"),
        ],
        capture_output=True,
        text=True,
        timeout=25,
        env={**__import__("os").environ, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"},
    )
    assert result.returncode == 0, result.stderr
