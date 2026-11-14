import gc
import json
import tracemalloc
from typing import Any, cast

from langgraph_instrumentation import (
    DeterministicClock,
    MemoryTraceStore,
    RecordingLimits,
    SpanKind,
    SpanStatus,
    TraceRecorder,
)


def test_per_trace_and_per_span_caps_preserve_structure():
    store = MemoryTraceStore()
    recorder = TraceRecorder(
        store,
        limits=RecordingLimits(
            max_trace_spans=2,
            max_span_events=2,
            max_trace_events=3,
            max_trace_metrics=1,
        ),
    )
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    child = recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="root")
    assert root and child
    assert (
        recorder.start_span("dropped", "node", SpanKind.NODE, parent_correlation_id="root") is None
    )
    for i in range(10):
        recorder.add_event("root", f"event-{i}")
        recorder.add_event("child", f"event-{i}")
        recorder.record_metric("root", "counter", {"count": i})
    recorder.end_span("child")
    recorder.end_span("root")
    bundle = store.get_trace(root.trace_id)
    assert bundle is not None
    assert len(bundle.spans) == 2
    assert sum(len(s.events) for s in bundle.spans) == 3
    assert all(len(s.events) <= 2 for s in bundle.spans)
    assert len(bundle.metrics) == 1
    assert recorder.diagnostics["events"] == 17
    assert recorder.close()


def test_byte_budgets_apply_across_payloads_and_roots():
    store = MemoryTraceStore()
    recorder = TraceRecorder(
        store,
        limits=RecordingLimits(
            max_total_attribute_bytes=200,
            max_trace_attribute_bytes=100,
        ),
    )
    for i in range(10):
        recorder.start_span(str(i), "root", SpanKind.GRAPH, attributes={"body": "x" * 80})
        recorder.end_span(str(i))
    assert recorder._attribute_bytes <= 200
    assert all(c["bytes"] <= 100 for c in recorder._trace_counts.values())
    assert recorder.diagnostics["bytes"] > 0
    assert recorder.close()
    assert not recorder._trace_counts


def test_active_cap_and_cooperative_expiration_reclaim_payloads():
    clock = DeterministicClock(unix_time_ns=100, monotonic_time_ns=100)
    store = MemoryTraceStore()
    recorder = TraceRecorder(
        store,
        clock=clock,
        limits=RecordingLimits(
            max_active_spans=2,
            max_span_age_ns=10,
        ),
    )
    root = recorder.start_span("root", "root", SpanKind.GRAPH, attributes={"payload": "value"})
    recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="root")
    assert recorder.start_span("denied", "denied", SpanKind.GRAPH) is None
    clock.advance(10)
    assert recorder.expire_spans() == 2
    assert not recorder._active
    assert recorder.end_span("root") is None  # late completion is harmless
    assert recorder.diagnostics["lifecycle"] == 0
    assert recorder.start_span("next", "next", SpanKind.GRAPH)
    assert root is not None
    bundle = store.get_trace(root.trace_id)
    assert bundle and all(s.status is SpanStatus.ABANDONED for s in bundle.spans)
    assert recorder.close()


def test_event_appends_do_not_rebuild_spans(monkeypatch):
    import langgraph_instrumentation.models as models

    store = MemoryTraceStore()
    recorder = TraceRecorder(store)
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    original = models.Span.__post_init__
    calls = []

    def checked(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(models.Span, "__post_init__", checked)
    for _ in range(100):
        recorder.add_event("root", "chunk")
    assert calls == []
    assert root is not None
    bundle = store.get_trace(root.trace_id)
    assert bundle is not None and len(bundle.spans[0].events) == 100
    assert calls
    assert recorder.close()


def test_dropped_pressure_does_not_retain_caller_payloads():
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    store = MemoryTraceStore(max_traces=2)
    recorder = TraceRecorder(
        store,
        limits=RecordingLimits(
            max_spans=4,
            max_events=10,
            max_span_events=10,
            max_total_attribute_bytes=4096,
            max_trace_attribute_bytes=2048,
        ),
    )
    recorder.start_span("root", "root", SpanKind.GRAPH)
    for _ in range(5_000):
        recorder.add_event("root", "event", attributes={"body": "x" * 100_000})
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert retained < 8_000_000  # broad allocator/coverage tolerance, not exact RSS
    assert recorder.diagnostics["events"] == 4990
    assert recorder.close()


def test_real_stream_summary_and_capped_detail(local_agent):
    from langgraph_instrumentation import CapturePolicy, RecordingSession

    with RecordingSession(
        capture=CapturePolicy.bounded(),
        capture_streaming_tokens=True,
        limits=RecordingLimits(max_span_events=1),
    ) as session:
        list(
            local_agent.stream(
                {"messages": [("user", "add")]},
                config={"callbacks": [session.handler]},
                stream_mode="messages",
            )
        )
    bundle = session.snapshot()[0]
    llms = [s for s in bundle.spans if s.kind is SpanKind.LLM]
    assert all(cast(Any, s.attributes["llm.streaming"])["chunks"] > 1 for s in llms)
    assert all(len(s.events) == 1 for s in llms)
    assert session.diagnostics["events"] > 0
    assert "first_chunk_offset_ns" in json.dumps(bundle.to_dict())
