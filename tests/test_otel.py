import io
import json
import subprocess
import sys
import threading
from dataclasses import replace

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from langchain_ollie import RecordingSession, SpanKind, SpanLink, SpanStatus
from langchain_ollie.exporters.otel import OpenTelemetryProcessor, to_otel_span

pytestmark = pytest.mark.otel


def test_actual_agent_sdk_parentage_resource_and_perfetto_unchanged(local_agent):
    exporter = InMemorySpanExporter()
    processor = OpenTelemetryProcessor(exporter, batch=False)
    with RecordingSession(processors=[processor], strict=True, service_name="otel-test") as session:
        result = local_agent.invoke(
            {"messages": [("user", "add")]}, config={"callbacks": [session.handler]}
        )
    assert result["messages"][-1].content == "Result: 5"
    local = {s.span_id.value: s for b in session.snapshot() for s in b.spans}
    spans = exporter.get_finished_spans()
    assert len(spans) == len(local)
    for span in spans:
        assert span.context is not None
        original = local[span.context.span_id]
        assert span.context.trace_id == original.trace_id.value
        assert (span.parent.span_id if span.parent else None) == (
            original.parent_span_id.value if original.parent_span_id else None
        )
        assert span.resource.attributes["service.name"] == "otel-test"
        assert span.start_time == original.start_time_unix_ns
        assert span.end_time == original.end_time_unix_ns
        assert span.attributes is not None
        assert span.attributes["local_trace.span.kind"] == original.kind.value
    output = io.StringIO()
    session.export(output)
    from langchain_ollie import PerfettoExporter

    assert json.loads(output.getvalue()) == PerfettoExporter().to_dict(
        session.snapshot(), diagnostics=session.diagnostics
    )


def test_explicit_links_events_status_and_export_only_context():
    # SDK current context is deliberately not grafted into local parentage.
    incoming = otel_trace.SpanContext(
        999, 111, True, otel_trace.TraceFlags(0), otel_trace.TraceState([("vendor", "value")])
    )
    with otel_trace.use_span(otel_trace.NonRecordingSpan(incoming)), RecordingSession() as session:
        root = session.recorder.start_span("root", "root", SpanKind.GRAPH)
        session.recorder.add_event("root", "observed", attributes={"count": 1})
        completed = session.recorder.end_span("root", status=SpanStatus.CANCELLED)
    assert completed is not None and root is not None
    converted = to_otel_span(
        replace(completed, links=(SpanLink(root.trace_id, root.span_id, "resumes"),))
    )
    assert converted.context is not None and converted.context.trace_id != incoming.trace_id
    assert converted.parent is None
    assert converted.context.trace_flags.sampled and not converted.context.is_remote
    assert not converted.context.trace_state
    assert converted.status.status_code is StatusCode.UNSET
    assert converted.links[0].context.span_id == root.span_id.value
    assert converted.links[0].attributes is not None
    assert converted.links[0].attributes["local_trace.relationship"] == "resumes"
    assert converted.events[0].name == "observed"
    assert converted.events[0].timestamp == completed.events[0].time_unix_ns
    with pytest.raises(ValueError, match="completed"):
        to_otel_span(root)


def test_export_failure_does_not_erase_local_trace_or_log_bodies(caplog):
    class Failing(SpanExporter):
        def export(self, spans):
            raise OSError("OTEL-PRIVATE-CANARY")

    bridge = OpenTelemetryProcessor(Failing(), batch=False)
    with RecordingSession(processors=[bridge]) as session:
        session.recorder.start_span("root", "root", SpanKind.GRAPH)
        session.recorder.end_span("root")
    assert session.snapshot()[0].spans
    assert bridge.diagnostics["export_failures"] == 1
    assert session.diagnostics["destination_failures"] > 0
    assert "OTEL-PRIVATE-CANARY" not in caplog.text


def test_batch_flush_and_bounded_shutdown_wait():
    entered, release = threading.Event(), threading.Event()

    class Held(InMemorySpanExporter):
        def shutdown(self):
            entered.set()
            assert release.wait(5)
            super().shutdown()

    exporter = Held()
    bridge = OpenTelemetryProcessor(exporter, max_queue_size=4, max_export_batch_size=2)
    with RecordingSession() as session:
        session.recorder.start_span("root", "root", SpanKind.GRAPH)
        completed = session.recorder.end_span("root")
    assert completed is not None
    bridge.on_end(completed)
    assert bridge.force_flush(timeout=5)
    assert len(exporter.get_finished_spans()) == 1
    try:
        assert not bridge.shutdown(timeout=0)
        assert entered.wait(5)
        assert not bridge.shutdown(timeout=0)
    finally:
        release.set()
    assert bridge.shutdown(timeout=5)
    assert bridge._shutdown_thread is not None and not bridge._shutdown_thread.is_alive()


def test_default_import_has_no_otel_sdk_or_bridge():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import langchain_ollie; "
                "assert 'opentelemetry.sdk' not in sys.modules; "
                "assert 'langchain_ollie.exporters.otel' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_flush_timeout_is_bounded_even_when_sdk_export_ignores_it():
    entered, release = threading.Event(), threading.Event()

    class Held(InMemorySpanExporter):
        def export(self, spans):
            entered.set()
            assert release.wait(5)
            return super().export(spans)

    bridge = OpenTelemetryProcessor(Held())
    with RecordingSession() as session:
        session.recorder.start_span("root", "root", SpanKind.GRAPH)
        completed = session.recorder.end_span("root")
    assert completed is not None
    bridge.on_end(completed)
    try:
        assert not bridge.force_flush(timeout=0.01)
        assert entered.wait(5)
        worker = bridge._flush_thread
        assert not bridge.force_flush(timeout=0)
        assert bridge._flush_thread is worker
    finally:
        release.set()
    assert bridge.shutdown(timeout=5)
