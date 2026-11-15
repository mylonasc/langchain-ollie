"""Privacy and finite-admission regressions."""

import json
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from langchain_ollie import (
    CapturePolicy,
    LangGraphCallbackHandler,
    MemoryTraceStore,
    RecordingLimits,
    RedactionPolicy,
    SpanKind,
    TraceRecorder,
)

CANARY = "CUSTOMER-SECRET-CANARY"


@pytest.mark.parametrize("policy", [CapturePolicy.none(), CapturePolicy.metadata()])
@pytest.mark.parametrize("callback", ["on_chain_error", "on_llm_error", "on_tool_error"])
def test_disabled_capture_never_persists_exception_bodies(policy, callback, caplog) -> None:
    store = MemoryTraceStore()
    recorder = TraceRecorder(store)
    handler = LangGraphCallbackHandler(recorder, capture=policy, capture_exception_messages=True)
    run_id = uuid4()
    handler.on_chain_start({}, {"input": CANARY}, run_id=run_id)
    getattr(handler, callback)(ValueError(CANARY), run_id=run_id)
    bundle = store.get_trace(store.list_traces()[0].trace_id)
    assert bundle is not None
    assert CANARY not in json.dumps(bundle.to_dict())
    assert CANARY not in caplog.text
    assert not bundle.trace.attributes
    assert all(not context.trace.attributes for context in recorder._completed_contexts.values())
    assert recorder.close()


def test_exception_messages_need_separate_opt_in_and_use_redaction() -> None:
    for opted_in in (False, True):
        store = MemoryTraceStore()
        recorder = TraceRecorder(store)
        handler = LangGraphCallbackHandler(
            recorder,
            capture=CapturePolicy.full(),
            capture_exception_messages=opted_in,
            redaction=RedactionPolicy(keys=frozenset({"exception.message"})),
        )
        run_id = uuid4()
        handler.on_chain_start({}, {}, run_id=run_id)
        handler.on_chain_error(ValueError(CANARY), run_id=run_id)
        bundle = store.get_trace(store.list_traces()[0].trace_id)
        assert bundle is not None
        assert CANARY not in json.dumps(bundle.to_dict())
        assert ("details" in cast(Any, bundle.spans[0].attributes["error"])) is opted_in
        assert recorder.close()


def test_store_and_processor_exceptions_do_not_enter_diagnostic_logs(caplog) -> None:
    class FailingStore(MemoryTraceStore):
        def start_span(self, trace, span):
            raise RuntimeError(CANARY)

    class FailingProcessor:
        def on_end(self, span):
            raise RuntimeError(CANARY)

        def force_flush(self, timeout=None):
            raise RuntimeError(CANARY)

        def shutdown(self, timeout=None):
            raise RuntimeError(CANARY)

    recorder = TraceRecorder(FailingStore())
    recorder.start_span("root", "graph", SpanKind.GRAPH)
    assert recorder.close()
    recorder = TraceRecorder(processors=[FailingProcessor()])
    recorder.start_span("root", "graph", SpanKind.GRAPH)
    recorder.end_span("root")
    assert not recorder.force_flush()
    assert not recorder.close()
    assert caplog.records
    assert CANARY not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_projection_failure_is_contained_before_framework_logging(caplog) -> None:
    def failing_redactor(path, key, value):
        raise RuntimeError(CANARY)

    recorder = TraceRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.full(),
        redaction=RedactionPolicy(predicate=failing_redactor),
    )
    handler.on_chain_start({}, {"body": CANARY}, run_id=uuid4())
    assert caplog.records
    assert CANARY not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert recorder.close()


def test_failed_write_does_not_consume_message_content() -> None:
    class FailOnce(MemoryTraceStore):
        failed = False

        def start_span(self, trace, span):
            if not self.failed:
                self.failed = True
                raise RuntimeError("synthetic failure")
            super().start_span(trace, span)

    store = FailOnce()
    recorder = TraceRecorder(store)
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.bounded())
    message = HumanMessage(content="retained", id="same-id")
    for _ in range(3):
        handler.on_chain_start({}, {"message": message}, run_id=uuid4())
    for summary in store.list_traces():
        bundle = store.get_trace(summary.trace_id)
        assert bundle is not None
        assert "retained" in json.dumps(bundle.to_dict())
        assert "repeated" not in json.dumps(bundle.to_dict())
    assert len(store.list_traces()) == 2
    assert recorder.close()


def test_metadata_summarizes_original_not_truncated_projection() -> None:
    store = MemoryTraceStore()
    recorder = TraceRecorder(store)
    handler = LangGraphCallbackHandler(recorder)
    handler.on_chain_start({}, {str(i): CANARY for i in range(1_000)}, run_id=uuid4())
    bundle = store.get_trace(store.list_traces()[0].trace_id)
    assert bundle is not None
    assert bundle.spans[0].attributes["input"] == {"type": "mapping", "size": 1_000}
    assert recorder.close()


def test_full_capture_does_not_apply_bounded_collection_limit() -> None:
    store = MemoryTraceStore()
    recorder = TraceRecorder(store)
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.full())
    handler.on_chain_start({}, {"items": list(range(100))}, run_id=uuid4())
    bundle = store.get_trace(store.list_traces()[0].trace_id)
    assert bundle is not None
    assert cast(Any, bundle.spans[0].attributes["input"])["items"] == tuple(range(100))
    assert recorder.close()


def test_global_capture_safety_limits_apply_to_full_and_broad_trees() -> None:
    policy = CapturePolicy.full()
    assert policy.capture("x" * 1_000_000) == "<payload-budget>"
    broad = {str(i): [str(j) for j in range(1_000)] for i in range(100)}
    result = policy.capture(broad)
    assert len(json.dumps(result).encode()) <= policy.max_payload_bytes
    assert "budget" in json.dumps(result) or "truncated" in json.dumps(result)
    deep: dict[str, object] = {}
    for _ in range(1_000):
        deep = {"nested": deep}
    assert "<max-depth>" in json.dumps(policy.capture(deep))


def test_recording_admission_is_finite_and_loss_is_visible() -> None:
    store = MemoryTraceStore()
    recorder = TraceRecorder(
        store,
        limits=RecordingLimits(max_spans=2, max_events=1, max_metrics=1, max_lanes=1),
    )
    assert recorder.start_span("root", "graph", SpanKind.GRAPH)
    assert recorder.start_span("child", "node", SpanKind.NODE, lane_id="different")
    assert recorder.start_span("dropped", "node", SpanKind.NODE) is None
    assert recorder.add_event("root", "kept")
    assert recorder.add_event("root", "dropped") is None
    assert recorder.record_metric("root", "usage", {"total": 1})
    assert recorder.record_metric("root", "usage", {"total": 2}) is None
    assert recorder.end_span("root", attributes={"body": "x" * 1_000_000})
    assert recorder.end_span("child")
    assert recorder.diagnostics == {
        "spans": 1,
        "events": 1,
        "metrics": 1,
        "attributes": 1,
        "lanes": 1,
        "destination_failures": 0,
        "lifecycle": 0,
        "bytes": 0,
        "active_spans": 0,
        "expired_spans": 0,
    }
    snapshot = recorder.diagnostics
    snapshot["spans"] = 999
    assert recorder.diagnostics["spans"] == 1
    assert recorder.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_recording_limits_are_rejected(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        RecordingLimits(max_spans=value)
