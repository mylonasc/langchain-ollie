"""Check shipped entrypoints against callback evidence and their known scripts.

Raw callback names are deliberately retained independently of span display names:
matching a span to another span would miss inherited-metadata classification bugs.
"""

import json
import runpy
import time
from collections import Counter
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from perfetto_support import assert_imported_spans, query

import langchain_ollie as library
from langchain_ollie import RecordingSession, SpanKind, SpanStatus
from langchain_ollie.adapters import LangGraphCallbackHandler

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "readme",
    "local_agent",
    "local_deep_agent",
    "local_sqlite",
    "local_resources",
    "local_otel",
)
HOOKS = (
    "on_chain_start",
    "on_chain_end",
    "on_chain_error",
    "on_chat_model_start",
    "on_llm_end",
    "on_llm_error",
    "on_tool_start",
    "on_tool_end",
    "on_tool_error",
)


@pytest.fixture(params=EXAMPLES)
def reviewed_example(request, monkeypatch, tmp_path):
    name = request.param
    if name == "local_resources":
        pytest.importorskip("psutil")
    sdk_exporters = []
    if name == "local_otel":
        module = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
        original_exporter = module.InMemorySpanExporter

        class ObservedExporter(original_exporter):
            def __init__(self):
                super().__init__()
                sdk_exporters.append(self)

        monkeypatch.setattr(module, "InMemorySpanExporter", ObservedExporter)
    sessions = []
    callbacks = []

    class ObservedSession(RecordingSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    monkeypatch.setattr(library, "RecordingSession", ObservedSession)

    def tap(hook, original):
        def callback(self, *args, **kwargs):
            serialized = args[0] if args and isinstance(args[0], dict) else {}
            record = {
                "hook": hook,
                "run_id": str(kwargs["run_id"]),
                "parent": str(kwargs["parent_run_id"]) if kwargs.get("parent_run_id") else None,
                "name": kwargs.get("name") or serialized.get("name"),
                "before": time.monotonic_ns(),
                "value": args[0] if args else None,
            }
            callbacks.append(record)
            result = original(self, *args, **kwargs)
            record["after"] = time.monotonic_ns()
            return result

        return callback

    for hook in HOOKS:
        monkeypatch.setattr(
            LangGraphCallbackHandler, hook, tap(hook, getattr(LangGraphCallbackHandler, hook))
        )
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    monkeypatch.chdir(tmp_path)
    if name == "readme":
        source = (ROOT / "README.md").read_text().split("```python\n", 1)[1].split("```", 1)[0]
        exec(compile(source, "README.md quickstart", "exec"), {})
    else:
        args = [name, "trace.json"]
        if name == "local_sqlite":
            args = [name, "recording.db", "trace.json"]
        monkeypatch.setattr("sys.argv", args)
        runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="__main__")
    assert len(sessions) == 1
    return name, sessions[0], callbacks, tmp_path / "trace.json", sdk_exporters


def test_example_behavior_matches_recorded_callbacks_and_artifact(reviewed_example):
    name, session, callbacks, artifact, sdk_exporters = reviewed_example
    bundles = session.snapshot()
    spans = [s for b in bundles for s in b.spans]
    starts = {c["run_id"]: c for c in callbacks if c["hook"].endswith("_start")}
    ends = {c["run_id"]: c for c in callbacks if c["hook"].endswith(("_end", "_error"))}
    assert starts.keys() == ends.keys()
    assert len(starts) == sum(c["hook"].endswith("_start") for c in callbacks)
    by_run = {
        s.attributes["langchain.run_id"]: s for s in spans if "langchain.run_id" in s.attributes
    }
    assert by_run.keys() == starts.keys()
    for run_id, start in starts.items():
        span, end = by_run[run_id], ends[run_id]
        assert start["before"] <= span.start_time_monotonic_ns <= start["after"]
        assert end["before"] <= span.end_time_monotonic_ns <= end["after"]
        assert span.parent_span_id == (by_run[start["parent"]].span_id if start["parent"] else None)
        if start["parent"]:
            parent = by_run[start["parent"]]
            assert parent.start_time_monotonic_ns <= span.start_time_monotonic_ns
            assert span.end_time_monotonic_ns <= parent.end_time_monotonic_ns
    assert all(s.status is SpanStatus.SUCCESS for s in spans)
    assert not any(session.diagnostics.values())

    operations = [
        c["name"] for c in callbacks if c["hook"] in ("on_chat_model_start", "on_tool_start")
    ]
    root = next(c for c in callbacks if c["hook"] == "on_chain_start" and c["parent"] is None)
    result = ends[root["run_id"]]["value"]
    if name == "readme":
        assert operations == ["FakeListChatModel"]
        assert result["messages"][-1].content == "Hello! I am a traced LangGraph agent."
        assert len(spans) == 3
    elif name == "local_deep_agent":
        assert operations == [
            "coordinator",
            "write_file",
            "coordinator",
            "read_file",
            "coordinator",
            "task",
            "worker",
            "multiply",
            "worker",
            "coordinator",
        ]
        assert result["files"]["/report.txt"]["content"] == "Local report"
        assert result["messages"][-1].content == (
            "Result: 10. Report written and read; worker completed."
        )
        assert len(spans) == 25
        task = next(c for c in callbacks if c["hook"] == "on_tool_start" and c["name"] == "task")
        delegated = next(c for c in callbacks if c["name"] == "general-purpose")
        assert delegated["parent"] == task["run_id"]
        assert Counter(c["name"] for c in starts.values())["model"] == 6
        display = Counter(s.name for s in spans)
        assert display["Node: model"] == 6
        assert display["Node: tools"] == 4
        assert display["Graph: general-purpose"] == 1
        subagent = next(s for s in spans if s.name == "Graph: general-purpose")
        assert subagent.kind is SpanKind.GRAPH
        assert all(
            s.name != "Node: tools" or s.attributes["langchain.run_id"] != delegated["run_id"]
            for s in spans
        )
    else:
        assert operations == ["ScriptedModel", "add", "ScriptedModel"]
        assert result["messages"][-1].content == "Result: 5"
        tools = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [(m.name, m.tool_call_id, m.content) for m in tools] == [("add", "call-add", "5")]
        assert len(spans) == (10 if name == "local_resources" else 9)
        # Two model-node executions, plus two routing callbacks that inherit the
        # node metadata. Routing must not inflate the node execution count.
        assert Counter(c["name"] for c in starts.values())["model"] == 2
        assert Counter(c["name"] for c in starts.values())["tools_condition"] == 2
        display = Counter(s.name for s in spans)
        assert display["Node: model"] == 2
        if name != "readme":
            assert display["tools_condition"] == 2

    models = [s for s in spans if s.kind.value == "llm"]
    expected_tokens = 0 if name == "readme" else 18 if name == "local_deep_agent" else 8
    assert sum(s.attributes["usage"]["total_tokens"] for s in models) == expected_tokens
    metrics = [m for b in bundles for m in b.metrics if m.name == "llm.token.usage"]
    assert len(metrics) == (0 if name == "readme" else 6 if name == "local_deep_agent" else 2)
    if metrics:
        assert max(m.values["total_tokens"] for m in metrics) == expected_tokens
    payload = json.loads(artifact.read_text())
    assert not payload["otherData"]["incomplete"]
    assert len([e for e in payload["traceEvents"] if e["ph"] == "X"]) == len(spans)
    completed = [
        (e["ts"], e["ts"] + e["dur"], e["tid"], e["name"])
        for e in payload["traceEvents"]
        if e["ph"] == "X"
    ]
    for left in completed:
        for right in completed:
            if left is not right and left[0] < right[1] and right[0] < left[1]:
                assert left[2] != right[2], f"overlapping slices share a track: {left} {right}"
    if name == "local_resources":
        process = next(s for s in spans if s.name == "Process resources")
        assert process.parent_span_id is None
        points = [m for b in bundles for m in b.metrics if m.name == "process.resources"]
        assert points and all(m.trace_id == process.trace_id for m in points)
        assert all(m.attributes["measurement.scope"] == "process" for m in points)
    if sdk_exporters:
        sdk = sdk_exporters[0].get_finished_spans()
        assert {s.context.span_id for s in sdk} == {s.span_id.value for s in spans}
        for exported in sdk:
            local = next(s for s in spans if s.span_id.value == exported.context.span_id)
            assert exported.start_time == local.start_time_unix_ns
            assert exported.end_time == local.end_time_unix_ns
            assert (exported.parent.span_id if exported.parent else None) == (
                local.parent_span_id.value if local.parent_span_id else None
            )


@pytest.mark.perfetto
def test_every_example_imports_with_matching_slices_flows_and_usage(
    reviewed_example, trace_processor
):
    name, session, _, artifact, _ = reviewed_example
    assert_imported_spans(trace_processor, artifact, session.snapshot())
    flows = query(
        trace_processor,
        artifact,
        "SELECT extract_arg(parent.arg_set_id,'args.span_id') AS parent_id, "
        "extract_arg(child.arg_set_id,'args.span_id') AS child_id "
        "FROM flow JOIN slice parent ON parent.id=flow.slice_out "
        "JOIN slice child ON child.id=flow.slice_in",
    )
    assert {(row["parent_id"], row["child_id"]) for row in flows} == {
        (str(s.parent_span_id), str(s.span_id))
        for bundle in session.snapshot()
        for s in bundle.spans
        if s.parent_span_id is not None
    }
    totals = query(
        trace_processor,
        artifact,
        "SELECT value FROM counter JOIN counter_track ON counter.track_id=counter_track.id "
        "WHERE counter_track.name='llm.token.usage total_tokens' ORDER BY ts",
    )
    expected = (
        [] if name == "readme" else [3, 6, 9, 12, 15, 18] if name == "local_deep_agent" else [3, 8]
    )
    assert [float(row["value"]) for row in totals] == expected
