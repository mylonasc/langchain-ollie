import asyncio
import operator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Annotated, Any, TypedDict, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt
from perfetto_support import assert_imported_spans, query

from langgraph_instrumentation import (
    MemoryTraceStore,
    PerfettoExporter,
    RecordingSession,
    SpanLink,
    SpanStatus,
)


class State(TypedDict, total=False):
    values: Annotated[list[int], operator.add]
    item: int
    count: int


def new_graph() -> Any:
    # ty cannot currently match TypedDict against LangGraph's structural bound.
    return StateGraph(cast(Any, State))


def invoke(app, inputs, config, asynchronous):
    return (
        asyncio.run(app.ainvoke(inputs, config=config))
        if asynchronous
        else app.invoke(inputs, config=config)
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dynamic_send_fanout_is_concurrent_and_not_fake_causality(asynchronous):
    barrier = Barrier(2)

    def worker(state):
        barrier.wait(timeout=5)
        return {"values": [state["item"]]}

    graph = new_graph()
    graph.add_node("worker", worker)
    graph.add_conditional_edges(START, lambda s: [Send("worker", {"item": i}) for i in (1, 2)])
    graph.add_edge("worker", END)
    with RecordingSession(strict=True) as session:
        result = invoke(
            graph.compile(), {"values": []}, {"callbacks": [session.handler]}, asynchronous
        )
    assert sorted(result["values"]) == [1, 2]
    bundle = session.snapshot()[0]
    workers = [s for s in bundle.spans if s.name == "Node: worker"]
    assert len(workers) == 2 and workers[0].parent_span_id == workers[1].parent_span_id
    assert max(s.start_time_monotonic_ns for s in workers) < min(
        s.end_time_monotonic_ns or 0 for s in workers
    )
    assert not any(s.links for s in bundle.spans)
    payload = PerfettoExporter().to_dict([bundle])
    events = [e for e in payload["traceEvents"] if e["name"] == "Node: worker" and e["ph"] == "X"]
    assert len({e["tid"] for e in events}) == 2


def nested_loop():
    inner = new_graph()
    inner.add_node("repeat", lambda state: {"count": state["count"] + 1})
    inner.add_edge(START, "repeat")
    inner.add_edge("repeat", END)
    outer = new_graph()
    outer.add_node("repeat", inner.compile(name="inner"))
    outer.add_edge(START, "repeat")
    outer.add_conditional_edges("repeat", lambda s: END if s["count"] >= 3 else "repeat")
    return outer.compile(name="outer")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_nested_repeated_nodes_and_loops(asynchronous):
    with RecordingSession(strict=True) as session:
        result = invoke(nested_loop(), {"count": 0}, {"callbacks": [session.handler]}, asynchronous)
    assert result["count"] == 3
    bundle = session.snapshot()[0]
    repeated = [s for s in bundle.spans if s.name == "Node: repeat"]
    assert len(repeated) >= 6
    assert len({s.span_id for s in repeated}) == len(repeated)
    assert not any(session.diagnostics.values())


@pytest.mark.parametrize("terminal", [False, True])
def test_retry_attempts_do_not_override_root_outcome(terminal):
    calls = []

    def work(state):
        calls.append(1)
        if terminal or len(calls) == 1:
            raise ValueError("synthetic retry")
        return {"count": 1}

    graph = new_graph()
    graph.add_node(
        "work",
        work,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0,
            jitter=False,
            retry_on=ValueError,
        ),
    )
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    store = MemoryTraceStore()
    with RecordingSession(store=store, strict=True) as session:
        if terminal:
            with pytest.raises(ValueError):
                graph.compile().invoke({"count": 0}, config={"callbacks": [session.handler]})
        else:
            assert (
                graph.compile().invoke({"count": 0}, config={"callbacks": [session.handler]})[
                    "count"
                ]
                == 1
            )
    assert len(calls) == 2
    assert any(s.status is SpanStatus.ERROR for s in session.snapshot()[0].spans)
    assert store.list_traces()[0].status is (SpanStatus.ERROR if terminal else SpanStatus.SUCCESS)
    store.close()


def resume_scenario():
    graph = new_graph()

    def approve(state):
        decision = interrupt("approval required")
        return {"count": 1 if decision else 0}

    graph.add_node("approve", approve)
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    app = graph.compile(checkpointer=InMemorySaver(), name="approval")
    session = RecordingSession(strict=True)
    cfg: RunnableConfig = {
        "callbacks": [session.handler],
        "configurable": {"thread_id": "approval-session"},
    }
    result = app.invoke({"count": 0}, config=cfg)
    assert result["__interrupt__"]
    previous = next(s for s in session.snapshot()[0].spans if s.parent_span_id is None)
    checkpoint = app.get_state(cfg).config["configurable"]
    assert previous.attributes["execution.interrupted"]
    link = SpanLink(previous.trace_id, previous.span_id, "resumes")
    resumed = app.invoke(
        Command(resume=True),
        config={
            "callbacks": [session.handler],
            "configurable": checkpoint,
            "metadata": {"local_trace.links": [link.to_dict()]},
        },
    )
    assert resumed["count"] == 1
    assert session.close()
    return session, link


def test_checkpoint_resume_is_linked_not_a_fake_continuous_span():
    session, link = resume_scenario()
    bundles = session.snapshot()
    assert len(bundles) == 2
    roots = [s for b in bundles for s in b.spans if s.parent_span_id is None]
    assert len({s.trace_id for s in roots}) == 2
    assert sum(link in s.links for s in roots) == 1
    assert all(s.status is SpanStatus.SUCCESS for s in roots)
    assert SpanLink.from_dict(link.to_dict()) == link


def test_two_invocations_share_handler_without_cross_parentage():
    barrier = Barrier(2)
    graph = new_graph()

    def work(state):
        barrier.wait(timeout=5)
        return {"count": state["count"] + 1}

    graph.add_node("work", work)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    app = graph.compile()
    with RecordingSession(strict=True) as session:

        def run(value):
            return app.invoke({"count": value}, config={"callbacks": [session.handler]})["count"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(run, [10, 20])) == [11, 21]
    assert len(session.snapshot()) == 2
    for bundle in session.snapshot():
        assert len([s for s in bundle.spans if s.parent_span_id is None]) == 1
        assert all(s.trace_id == bundle.trace.trace_id for s in bundle.spans)
    assert not any(session.diagnostics.values())


def test_parallel_deepagent_workers_have_distinct_task_ancestors(deep_agent_example):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgent

    barrier = Barrier(2)

    @tool("multiply")
    def multiply(a: int, b: int) -> int:
        """Synchronize two local multiplication workers."""
        barrier.wait(timeout=5)
        return a * b

    reply = deep_agent_example["_reply"]
    model = deep_agent_example["ScriptedModel"]
    first = reply(
        name="task",
        args={"subagent_type": "general-purpose", "description": "multiply"},
        call_id="a",
    )
    first.tool_calls.append(
        {
            "name": "task",
            "args": {"subagent_type": "worker-b", "description": "multiply"},
            "id": "b",
            "type": "tool_call",
        }
    )
    coordinator = model(name="coordinator", responses=(first, reply(content="done")))
    workers = [
        model(
            name=name,
            responses=(
                reply(name="multiply", args={"a": 2, "b": 5}, call_id="multiply"),
                reply(content="10"),
            ),
        )
        for name in ("worker-a", "worker-b")
    ]
    specs: list[SubAgent] = [
        {
            "name": name,
            "description": "local worker",
            "model": worker,
            "tools": [multiply],
            "middleware": [cast(Any, FilesystemMiddleware(tools=["read_file"]))],
        }
        for name, worker in zip(("general-purpose", "worker-b"), workers, strict=True)
    ]
    app = create_deep_agent(
        model=coordinator,
        backend=StateBackend(),
        middleware=[FilesystemMiddleware(tools=["read_file"])],
        subagents=specs,
    )
    with RecordingSession(strict=True) as session:
        result = app.invoke(
            {"messages": [("user", "multiply twice")]}, config={"callbacks": [session.handler]}
        )
    assert result["messages"][-1].content == "done"
    assert all(w.calls == 2 for w in workers)
    bundle = session.snapshot()[0]
    tools = [s for s in bundle.spans if s.name == "Tool: multiply"]
    assert len(tools) == 2
    by_id = {span.span_id: span for span in bundle.spans}
    task_ids = set()
    for span in tools:
        while span.parent_span_id is not None:
            span = by_id[span.parent_span_id]
            if span.name == "Tool: task":
                task_ids.add(span.span_id)
                break
    assert len(task_ids) == 2
    assert max(s.start_time_monotonic_ns for s in tools) < min(
        s.end_time_monotonic_ns or 0 for s in tools
    )
    assert not any(session.diagnostics.values())


@pytest.mark.perfetto
def test_imported_resume_links_and_nested_track_layout(trace_processor, tmp_path):
    session, _ = resume_scenario()
    path = tmp_path / "resume.json"
    session.export(path)
    artifact = PerfettoExporter().to_dict(session.snapshot())
    assert len([e for e in artifact["traceEvents"] if e.get("cat") == "causal_link"]) == 2
    assert_imported_spans(trace_processor, path, session.snapshot())
    with RecordingSession(strict=True) as nested:
        nested_loop().invoke({"count": 0}, config={"callbacks": [nested.handler]})
    nested.export(path)
    assert_imported_spans(trace_processor, path, nested.snapshot())
    overlaps = query(
        trace_processor,
        path,
        (
            "SELECT count(*) AS count FROM slice a JOIN slice b ON a.id < b.id "
            "AND a.track_id=b.track_id AND a.ts < b.ts+b.dur AND b.ts < a.ts+a.dur "
            "WHERE a.dur>0 AND b.dur>0"
        ),
    )
    assert int(overlaps[0]["count"]) == 0
