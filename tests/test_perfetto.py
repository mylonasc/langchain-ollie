import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from langgraph_instrumentation import (
    DeterministicClock,
    DeterministicIdGenerator,
    MemoryTraceStore,
    PerfettoExporter,
    RecordingLimits,
    RecordingSession,
    SpanKind,
    SpanStatus,
    TraceRecorder,
)


def bundle_fixture():
    clock = DeterministicClock(unix_time_ns=1_000_000, monotonic_time_ns=100_000)
    store = MemoryTraceStore()
    recorder = TraceRecorder(store, clock=clock, id_generator=DeterministicIdGenerator())
    root = recorder.start_span("root", "graph", SpanKind.GRAPH)
    assert root is not None
    clock.advance(1_000)
    recorder.start_span("left", "node", SpanKind.NODE, parent_correlation_id="root")
    recorder.start_span("right", "node", SpanKind.NODE, parent_correlation_id="root")
    clock.advance(2_000)
    recorder.add_event("left", "event", attributes={"step": 1})
    recorder.record_metric("left", "usage", {"tokens": 5})
    recorder.end_span("left")
    recorder.end_span("right", status=SpanStatus.ERROR)
    clock.advance(1_000)
    recorder.end_span("root")
    bundle = store.get_trace(root.trace_id)
    assert bundle is not None
    recorder.close()
    return bundle


def test_export_spans_events_counters_flows_and_overlap_safe_tracks() -> None:
    payload = PerfettoExporter().to_dict([bundle_fixture()], diagnostics={"events": 1})
    events = payload["traceEvents"]
    completed = [e for e in events if e["ph"] == "X"]
    assert [(e["name"], e["dur"]) for e in completed] == [("graph", 4), ("node", 2), ("node", 2)]
    assert len({e["tid"] for e in completed}) == 3
    assert len([e for e in events if e.get("cat") == "containment"]) == 4
    assert any(e["ph"] == "i" and e["name"] == "event" for e in events)
    assert any(e["ph"] == "C" and e["args"] == {"tokens": 5} for e in events)
    assert payload["otherData"]["incomplete"]


def test_deterministic_compact_output_and_atomic_failure(tmp_path, monkeypatch) -> None:
    exporter = PerfettoExporter()
    bundles = [bundle_fixture()]
    first, second = io.StringIO(), io.StringIO()
    exporter.export(bundles, first)
    exporter.export(bundles, second)
    assert first.getvalue() == second.getvalue()
    destination = tmp_path / "trace.json"
    destination.write_text("original")

    def fail(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("langgraph_instrumentation.exporters.perfetto.os.replace", fail)
    with pytest.raises(OSError):
        exporter.export(bundles, destination)
    assert destination.read_text() == "original"
    assert list(tmp_path.iterdir()) == [destination]


def test_export_rejects_mixed_clock_domains_before_writing() -> None:
    bundle = bundle_fixture()
    a = replace(bundle, trace=replace(bundle.trace, resource_attributes={"process.clock.id": "a"}))
    b = replace(bundle, trace=replace(bundle.trace, resource_attributes={"process.clock.id": "b"}))
    output = io.StringIO()
    with pytest.raises(ValueError, match="clock domains"):
        PerfettoExporter().export([a, b], output)
    assert not output.getvalue()


def test_session_exports_after_close_and_marks_abandoned_work(tmp_path) -> None:
    with RecordingSession() as session:
        session.recorder.start_span("root", "unfinished", SpanKind.GRAPH)
    assert session.close()
    bundle = session.snapshot()[0]
    assert bundle.spans[0].status is SpanStatus.ABANDONED
    destination = tmp_path / "trace.json"
    session.export(destination)
    payload = json.loads(destination.read_text())
    assert payload["otherData"]["recording_schema_version"] == 1
    assert payload["otherData"]["clock_domain"]
    assert payload["otherData"]["incomplete"]
    assert payload["traceEvents"]


def test_injected_store_is_not_closed_and_must_be_empty() -> None:
    store = MemoryTraceStore()
    with RecordingSession(store=store) as session:
        session.recorder.start_span("root", "graph", SpanKind.GRAPH)
        session.recorder.end_span("root")
    assert len(store.list_traces()) == 1
    with pytest.raises(ValueError, match="empty dedicated"):
        RecordingSession(store=store)
    assert store.close()


def test_budget_loss_and_active_work_are_visible_in_artifact() -> None:
    with RecordingSession(limits=RecordingLimits(max_spans=1)) as session:
        session.recorder.start_span("root", "graph", SpanKind.GRAPH)
        session.recorder.start_span("dropped", "graph", SpanKind.GRAPH)
        payload = PerfettoExporter().to_dict(
            session.snapshot(), diagnostics=session.recorder.diagnostics
        )
        assert payload["otherData"]["incomplete"]
        assert payload["otherData"]["loss"]["spans"] == 1
        assert any(e["ph"] == "i" for e in payload["traceEvents"])


def test_session_preserves_application_error() -> None:
    with pytest.raises(ValueError, match="application"), RecordingSession():
        raise ValueError("application")


def test_retention_loss_survives_session_close() -> None:
    store = MemoryTraceStore(max_traces=1)
    with RecordingSession(store=store) as session:
        for name in ("first", "second"):
            session.recorder.start_span(name, name, SpanKind.GRAPH)
            session.recorder.end_span(name)
    assert len(session.snapshot()) == 1
    assert session.diagnostics["store.evicted_traces"] == 1
    output = io.StringIO()
    session.export(output)
    assert json.loads(output.getvalue())["otherData"]["incomplete"]
    store.close()


def test_export_preserves_a_span_across_callback_threads() -> None:
    from uuid import uuid4

    run_id = uuid4()
    with RecordingSession(strict=True) as session:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                session.handler.on_chain_start,
                {},
                {},
                run_id=run_id,
                name="cross-thread",
                metadata={"ls_integration": "langgraph"},
            ).result(timeout=5)
        session.handler.on_chain_end({}, run_id=run_id)
    output = io.StringIO()
    session.export(output)
    completed = [e for e in json.loads(output.getvalue())["traceEvents"] if e["ph"] == "X"]
    assert len(completed) == 1
    assert completed[0]["name"] == "Graph: cross-thread"


def test_non_strict_session_does_not_replace_success_with_recording_failure(caplog) -> None:
    class FailingStore(MemoryTraceStore):
        fail = True

        def complete_span(self, span):
            if self.fail:
                raise OSError("PRIVATE-CANARY")
            super().complete_span(span)

    store = FailingStore()
    with RecordingSession(store=store) as session:
        session.recorder.start_span("root", "graph", SpanKind.GRAPH)
    assert session.diagnostics["destination_failures"] > 0
    output = io.StringIO()
    session.export(output)
    assert json.loads(output.getvalue())["otherData"]["incomplete"]
    assert "PRIVATE-CANARY" not in caplog.text
    store.fail = False
    assert session.close()
    store.close()
