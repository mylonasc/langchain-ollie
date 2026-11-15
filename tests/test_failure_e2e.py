import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.tools import tool

from langchain_ollie import (
    MemoryTraceStore,
    RecordingLimits,
    RecordingSession,
    SpanKind,
    SpanStatus,
    TraceRecorder,
)


class Collector:
    def __init__(self):
        self.spans = []

    def on_end(self, span):
        self.spans.append(span)

    def force_flush(self, timeout=None):
        return True

    def shutdown(self, timeout=None):
        return True


@pytest.mark.parametrize("operation", ["start", "event", "metric", "complete"])
def test_real_agent_destination_failure_does_not_cascade(local_agent, operation, caplog):
    class FailOnce(MemoryTraceStore):
        failed = False

        def fail(self, name):
            if name == operation and not self.failed:
                self.failed = True
                raise OSError("DESTINATION-PRIVATE-CANARY")

        def start_span(self, trace, span):
            self.fail("start")
            super().start_span(trace, span)

        def record_event(self, event):
            self.fail("event")
            super().record_event(event)

        def record_metric(self, metric):
            self.fail("metric")
            super().record_metric(metric)

        def complete_span(self, span):
            self.fail("complete")
            super().complete_span(span)

    store = FailOnce()
    collector = Collector()
    with RecordingSession(
        store=store,
        processors=[collector],
        capture_streaming_tokens=True,
        service_name="failure-test",
    ) as session:
        final = None
        for kind, value in local_agent.stream(
            {"messages": [("user", "add")]},
            config={"callbacks": [session.handler]},
            stream_mode=["messages", "values"],
        ):
            if kind == "values":
                final = value
    assert final is not None and final["messages"][-1].content == "Result: 5"
    assert store.failed
    assert session.diagnostics["destination_failures"] >= 1
    assert session.diagnostics["lifecycle"] == 0
    assert len([s for s in collector.spans if s.kind is SpanKind.LLM]) == 2
    assert all(s.resource_attributes["service.name"] == "failure-test" for s in collector.spans)
    assert len({s.trace_id for s in collector.spans}) == 1
    assert not session.recorder._active
    output = io.StringIO()
    session.export(output)
    assert json.loads(output.getvalue())["otherData"]["incomplete"]
    assert "DESTINATION-PRIVATE-CANARY" not in caplog.text
    store.close()


def test_export_processor_failure_isolated_and_resources_closed(local_agent, caplog):
    class FailingExporter(Collector):
        def on_end(self, span):
            raise OSError("EXPORT-PRIVATE-CANARY")

    collector = Collector()
    with RecordingSession(processors=[FailingExporter(), collector]) as session:
        result = local_agent.invoke(
            {"messages": [("user", "add")]}, config={"callbacks": [session.handler]}
        )
    assert result["messages"][-1].content == "Result: 5"
    assert collector.spans
    assert session.recorder.is_closed
    assert "EXPORT-PRIVATE-CANARY" not in caplog.text
    assert session.diagnostics["destination_failures"] > 0


def test_slow_store_wait_timeout_preserves_agent_and_eventually_closes(local_agent):
    entered, release = threading.Event(), threading.Event()

    class SlowStore(MemoryTraceStore):
        def start_span(self, trace, span):
            entered.set()
            assert release.wait(5)
            super().start_span(trace, span)

    store = SlowStore()
    session = RecordingSession(store=store)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            local_agent.invoke,
            {"messages": [("user", "add")]},
            config={"callbacks": [session.handler]},
        )
        try:
            assert entered.wait(5)
            assert session.close(timeout=0.01) is False
        finally:
            release.set()
        assert future.result(timeout=5)["messages"][-1].content == "Result: 5"
    assert session.close(timeout=5)
    assert session.recorder.is_closed and not session.recorder._active
    store.close()


def test_concurrent_session_close_is_idempotent(local_agent):
    session = RecordingSession(strict=True)
    local_agent.invoke({"messages": [("user", "add")]}, config={"callbacks": [session.handler]})
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(lambda _: session.close(timeout=5), range(16)))
    assert session.snapshot()
    assert not session.recorder._operations


@pytest.mark.parametrize("asynchronous", [False, True])
def test_early_deepagent_stream_close_preserves_consumer_control(deep_agent_example, asynchronous):
    scenario = deep_agent_example["build_deep_agent"]()
    with RecordingSession(strict=True) as session:
        cfg = {"callbacks": [session.handler]}
        inputs = {"messages": [("user", "begin")]}
        if asynchronous:

            async def consume_one():
                before = asyncio.all_tasks()
                stream = scenario.agent.astream(inputs, config=cfg, stream_mode="messages")
                await anext(stream)
                await stream.aclose()
                assert not (asyncio.all_tasks() - before)

            asyncio.run(consume_one())
        else:
            stream = scenario.agent.stream(inputs, config=cfg, stream_mode="messages")
            next(stream)
            stream.close()
    assert not session.recorder._active
    statuses = {s.status for b in session.snapshot() for s in b.spans}
    assert statuses & {SpanStatus.CANCELLED, SpanStatus.ABANDONED}


def test_cancellation_during_real_deepagent_tool(deep_agent_example):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        @tool("multiply")
        async def held(a: int, b: int) -> int:
            """Wait cooperatively before local multiplication."""
            entered.set()
            await release.wait()
            return a * b

        scenario = deep_agent_example["build_deep_agent"](worker_tool=held)
        with RecordingSession(strict=True) as session:
            task = asyncio.create_task(
                scenario.agent.ainvoke(
                    {"messages": [("user", "begin")]}, config={"callbacks": [session.handler]}
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
        assert not session.recorder._active
        assert any(s.status is SpanStatus.CANCELLED for b in session.snapshot() for s in b.spans)

    asyncio.run(run())


def test_pending_operations_are_bounded_without_blocking_caller(monkeypatch):
    entered, release, queued = threading.Event(), threading.Event(), threading.Event()

    class SlowStore(MemoryTraceStore):
        def start_span(self, trace, span):
            entered.set()
            assert release.wait(5)
            super().start_span(trace, span)

    recorder = TraceRecorder(SlowStore(), limits=RecordingLimits(max_pending_operations=1))
    original_wait = recorder._condition.wait

    def observed_wait(timeout=None):
        if recorder._operations:
            queued.set()
        return original_wait(timeout)

    monkeypatch.setattr(recorder._condition, "wait", observed_wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        start = pool.submit(recorder.start_span, "root", "root", SpanKind.GRAPH)
        try:
            assert entered.wait(5)
            end = pool.submit(recorder.end_span, "root")
            assert queued.wait(5)
            assert recorder.add_event("root", "denied") is None
            assert recorder.diagnostics["pending_operations"] == 1
        finally:
            release.set()
        assert start.result(timeout=5)
        assert end.result(timeout=5)
    assert recorder.close()


def test_session_finalization_reentry_is_rejected_without_deadlock(monkeypatch):
    session = RecordingSession()
    original = session._store.close

    def close(timeout=None):
        assert session.close(timeout=0) is False
        return original(timeout)

    monkeypatch.setattr(session._store, "close", close)
    assert session.close()


def test_exact_store_replays_are_idempotent():
    from dataclasses import replace

    store = MemoryTraceStore()
    with RecordingSession(store=store) as session:
        active = session.recorder.start_span("root", "root", SpanKind.GRAPH)
        completed = session.recorder.end_span("root")
    bundle = session.snapshot()[0]
    assert active is not None and completed is not None
    store.start_span(replace(bundle.trace, end_time_unix_ns=None), active)
    store.complete_span(completed)
    assert store.get_trace(bundle.trace.trace_id) == bundle
    store.close()
