import operator
from typing import Annotated, Any, TypedDict, cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from langgraph_instrumentation import (
    CapturePolicy,
    DeterministicClock,
    DeterministicIdGenerator,
    LangGraphCallbackHandler,
    MemoryTraceStore,
    SpanKind,
    TraceRecorder,
)
from langgraph_instrumentation.models import Span, TraceBundle


class AgentState(TypedDict):
    values: Annotated[list[str], operator.add]


@tool
def uppercase(value: str) -> str:
    """Convert a value to uppercase."""
    return value.upper()


def _semantic_trace(bundle: TraceBundle) -> set[tuple[str, SpanKind, str | None]]:
    names = {span.span_id: span.name for span in bundle.spans}
    return {(span.name, span.kind, names.get(span.parent_span_id)) for span in bundle.spans}


def _record_graph(*, asynchronous: bool) -> TraceBundle:
    model = FakeListChatModel(responses=["model-result"])

    def call_model(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
        response = model.invoke("prompt", config=config)
        return {"values": [str(response.content)]}

    def call_tool(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
        result = uppercase.invoke({"value": state["values"][-1]}, config=config)
        return {"values": [result]}

    # ty does not currently recognize TypedDict as satisfying LangGraph's StateLike bound.
    graph = StateGraph(cast(Any, AgentState))
    graph.add_node("model", call_model)
    graph.add_node("tool", call_tool)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tool")
    graph.add_edge("tool", END)
    app = graph.compile(name="AgentGraph")

    store = MemoryTraceStore()
    recorder = TraceRecorder(
        store,
        clock=DeterministicClock(unix_time_ns=100, monotonic_time_ns=100),
        id_generator=DeterministicIdGenerator(),
        strict=True,
    )
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.none(), raise_error=True)
    config: RunnableConfig = {
        "callbacks": [handler],
        "configurable": {"thread_id": "integration"},
    }
    if asynchronous:
        asyncio_run(app.ainvoke({"values": ["start"]}, config=config))
    else:
        app.invoke({"values": ["start"]}, config=config)

    summaries = store.list_traces()
    assert len(summaries) == 1
    bundle = store.get_trace(summaries[0].trace_id)
    assert bundle is not None
    return bundle


def asyncio_run(awaitable: Any) -> Any:
    import asyncio

    return asyncio.run(awaitable)


def test_real_langgraph_sync_and_async_have_equivalent_semantics() -> None:
    synchronous = _record_graph(asynchronous=False)
    asynchronous = _record_graph(asynchronous=True)

    expected = {
        ("Graph: AgentGraph", SpanKind.GRAPH, None),
        ("Node: model", SpanKind.NODE, "Graph: AgentGraph"),
        ("LLM: FakeListChatModel", SpanKind.LLM, "Node: model"),
        ("Node: tool", SpanKind.NODE, "Graph: AgentGraph"),
        ("Tool: uppercase", SpanKind.TOOL, "Node: tool"),
    }
    assert _semantic_trace(synchronous) == expected
    assert _semantic_trace(asynchronous) == expected


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_langgraph_fanout_and_fanin_are_siblings(asynchronous: bool) -> None:
    def branch(name: str):
        def run(state: AgentState) -> dict[str, list[str]]:
            return {"values": [name]}

        return run

    graph = StateGraph(cast(Any, AgentState))
    graph.add_node("left", branch("left"))
    graph.add_node("right", branch("right"))
    graph.add_node("join", branch("joined"))
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge(["left", "right"], "join")
    graph.add_edge("join", END)
    app = graph.compile(name="FanGraph")

    store = MemoryTraceStore()
    recorder = TraceRecorder(store, strict=True)
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.none(), raise_error=True)
    config: RunnableConfig = {"callbacks": [handler]}
    if asynchronous:
        asyncio_run(app.ainvoke({"values": []}, config=config))
    else:
        app.invoke({"values": []}, config=config)

    summary = store.list_traces()[0]
    bundle = store.get_trace(summary.trace_id)
    assert bundle is not None
    by_name: dict[str, Span] = {span.name: span for span in bundle.spans}
    root = by_name["Graph: FanGraph"]
    assert by_name["Node: left"].parent_span_id == root.span_id
    assert by_name["Node: right"].parent_span_id == root.span_id
    assert by_name["Node: join"].parent_span_id == root.span_id
    assert by_name["Node: left"].execution_lane != by_name["Node: right"].execution_lane
