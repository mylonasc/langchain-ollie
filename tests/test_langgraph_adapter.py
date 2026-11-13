import asyncio
from collections.abc import Iterator, Mapping
from concurrent.futures import CancelledError
from typing import Any, cast
from uuid import UUID

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langgraph_instrumentation import (
    CapturePolicy,
    LangGraphCallbackHandler,
    SpanKind,
    SpanStatus,
)


class FakeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def start_span(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("start_span", args, kwargs))
        return object()

    def add_event(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("add_event", args, kwargs))
        return object()

    def record_metric(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("record_metric", args, kwargs))
        return object()

    def end_span(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("end_span", args, kwargs))
        return object()


def _call(
    recorder: FakeRecorder,
    operation: str,
) -> tuple[tuple[object, ...], dict[str, object]]:
    matching = [(args, kwargs) for name, args, kwargs in recorder.calls if name == operation]
    assert matching
    return matching[-1]


def test_chain_callbacks_normalize_graph_node_ids_metadata_and_payloads() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.bounded())
    root_id = UUID("AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA")
    node_id = UUID("BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB")

    handler.on_chain_start(
        {},
        {"password": "secret", "value": 1},
        run_id=root_id,
        metadata={"ls_integration": "langgraph", "thread_id": "thread-1"},
        name="workflow",
    )
    handler.on_chain_start(
        {},
        {"value": 2},
        run_id=node_id,
        parent_run_id=root_id,
        metadata={"langgraph_node": "plan", "langgraph_step": 1},
        name="ignored",
    )

    root = recorder.calls[0]
    assert root[1] == (str(root_id), "Graph: workflow", SpanKind.GRAPH)
    assert root[2]["trace_name"] == "Graph: workflow"
    assert root[2]["attributes"] == {
        "langchain.run_id": str(root_id),
        "langgraph.metadata": {"thread_id": "thread-1"},
        "input": {"password": "<redacted>", "value": 1},
    }

    node = recorder.calls[1]
    assert node[1] == (str(node_id), "Node: plan", SpanKind.NODE)
    assert node[2]["parent_correlation_id"] == str(root_id)
    assert node[2]["lane_id"] == "langgraph.node:plan"
    assert node[2]["lane_name"] == "plan"
    assert node[2]["attributes"] == {
        "langchain.run_id": str(node_id),
        "langchain.parent_run_id": str(root_id),
        "langgraph.metadata": {"langgraph_node": "plan", "langgraph_step": 1},
        "input": {"value": 2},
    }


def test_model_tool_retriever_usage_streaming_and_cancellation_translation() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.bounded(),
        capture_streaming_tokens=True,
    )
    parent_id = UUID(int=1)
    model_id = UUID(int=2)
    tool_id = UUID(int=3)
    retriever_id = UUID(int=4)

    handler.on_chat_model_start(
        {"id": ["provider", "FakeChat"]},
        [[HumanMessage(content="hello")]],
        run_id=model_id,
        parent_run_id=parent_id,
        invocation_params={"model": "test-model"},
    )
    handler.on_llm_new_token("x", run_id=model_id)
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="answer"))]],
            llm_output={"token_usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ),
        run_id=model_id,
    )
    handler.on_tool_start(
        {"name": "lookup"},
        "raw",
        run_id=tool_id,
        parent_run_id=parent_id,
        inputs={"query": "safe", "api_key": "hidden"},
        tool_call_id="call-1",
    )
    handler.on_tool_end(AIMessage(content="result"), run_id=tool_id)
    handler.on_retriever_start(
        {"name": "docs"},
        "question",
        run_id=retriever_id,
        parent_run_id=parent_id,
    )
    handler.on_retriever_end(
        [Document(page_content="body", metadata={"token": "hidden"}, id="doc-1")],
        run_id=retriever_id,
    )
    handler.on_tool_error(CancelledError(), run_id=tool_id)
    handler.on_chain_error(asyncio.CancelledError(), run_id=parent_id)

    model_start = recorder.calls[0]
    assert model_start[1][1:] == ("LLM: test-model", SpanKind.LLM)
    assert model_start[2]["attributes"] == {
        "langchain.run_id": str(model_id),
        "langchain.parent_run_id": str(parent_id),
        "model": "test-model",
        "messages": [[{"type": "human", "content": "hello"}]],
    }
    _, token_event = _call(recorder, "add_event")
    assert token_event == {"attributes": {"token": "x"}}
    metric_args, metric_kwargs = _call(recorder, "record_metric")
    assert metric_args == (
        str(model_id),
        "llm.token.usage",
        {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    assert metric_kwargs == {"cumulative": True}

    tool_start = next(
        call for call in recorder.calls if call[0] == "start_span" and call[1][0] == str(tool_id)
    )
    assert tool_start[2]["attributes"] == {
        "langchain.run_id": str(tool_id),
        "langchain.parent_run_id": str(parent_id),
        "input": {"query": "safe", "api_key": "<redacted>"},
        "tool_call_id": "call-1",
    }
    retriever_end = [
        call for call in recorder.calls if call[0] == "end_span" and call[1][0] == str(retriever_id)
    ][0]
    assert retriever_end[2]["attributes"] == {
        "documents": [{"page_content": "body", "metadata": {"token": "<redacted>"}, "id": "doc-1"}]
    }
    cancelled = [
        call
        for call in recorder.calls
        if call[0] == "end_span" and call[2].get("status") is SpanStatus.CANCELLED
    ]
    assert [call[1][0] for call in cancelled] == [str(tool_id), str(parent_id)]


def test_streaming_is_disabled_by_default_and_handler_has_no_async_hooks() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(recorder)

    handler.on_llm_new_token("secret", run_id=UUID(int=1))

    assert recorder.calls == []
    assert not any(name.startswith("aon_") for name in LangGraphCallbackHandler.__dict__)


def test_metadata_only_capture_retains_model_and_tool_ids_without_payloads() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(recorder)

    handler.on_llm_start(
        {},
        ["sensitive prompt"],
        run_id=UUID(int=1),
        invocation_params={"model": "model-id"},
    )
    handler.on_tool_start(
        {"name": "lookup"},
        "sensitive input",
        run_id=UUID(int=2),
        tool_call_id="call-id",
    )

    llm_attributes = recorder.calls[0][2]["attributes"]
    tool_attributes = recorder.calls[1][2]["attributes"]
    assert llm_attributes == {
        "langchain.run_id": str(UUID(int=1)),
        "model": "model-id",
        "prompts": {"type": "list", "size": 1},
    }
    assert tool_attributes == {
        "langchain.run_id": str(UUID(int=2)),
        "input": {"type": "str", "length": 15},
        "tool_call_id": "call-id",
    }


def test_error_callback_does_not_conceal_application_error() -> None:
    class FailingRecorder(FakeRecorder):
        def end_span(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("instrumentation failed")

    handler = LangGraphCallbackHandler(FailingRecorder(), raise_error=True)

    handler.on_tool_error(ValueError("application failed"), run_id=UUID(int=1))


def test_error_callback_avoids_hostile_exception_formatters() -> None:
    class HostileError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("must not format")

        def __repr__(self) -> str:
            raise RuntimeError("must not format")

    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(recorder, capture=CapturePolicy.bounded())

    handler.on_chain_error(HostileError("safe argument", object()), run_id=UUID(int=1))

    args, kwargs = _call(recorder, "end_span")
    assert args == (str(UUID(int=1)),)
    assert kwargs["status"] is SpanStatus.ERROR
    assert kwargs["attributes"] == {
        "error": {
            "type": f"{HostileError.__module__}.{HostileError.__qualname__}",
        }
    }


def test_error_callback_preserves_application_error_when_recorder_raises_base_exception() -> None:
    class CancellingRecorder(FakeRecorder):
        def end_span(self, *args: object, **kwargs: object) -> object:
            raise asyncio.CancelledError

    handler = LangGraphCallbackHandler(CancellingRecorder(), raise_error=True)

    handler.on_llm_error(ValueError("application failed"), run_id=UUID(int=1))


def test_non_error_callbacks_do_not_swallow_process_control_exceptions() -> None:
    class InterruptingRecorder(FakeRecorder):
        def start_span(self, *args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

    handler = LangGraphCallbackHandler(InterruptingRecorder())

    with pytest.raises(KeyboardInterrupt):
        handler.on_chain_start({}, {}, run_id=UUID(int=1), name="chain")


def test_callback_names_and_identifiers_are_bounded_without_formatting_objects() -> None:
    class HostileValue:
        def __str__(self) -> str:
            raise RuntimeError("must not format")

        def __repr__(self) -> str:
            raise RuntimeError("must not format")

    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.bounded(max_string_length=12),
    )
    hostile = HostileValue()

    handler.on_chain_start(
        {},
        {},
        run_id=UUID(int=1),
        metadata={"langgraph_node": hostile},
        name=hostile,
    )
    handler.on_chat_model_start(
        {"name": hostile},
        [[]],
        run_id=UUID(int=2),
        invocation_params={"model": hostile},
    )
    handler.on_tool_start(
        {"name": "x" * 100},
        "input",
        run_id=UUID(int=3),
        tool_call_id=hostile,
    )
    handler.on_retriever_start(
        {"name": "y" * 100},
        "query",
        run_id=UUID(int=4),
    )
    handler.on_tool_start(
        {"name": "safe"},
        "input",
        run_id=UUID(int=5),
        tool_call_id="z" * 100,
    )

    starts = [call for call in recorder.calls if call[0] == "start_span"]
    assert [call[1][1] for call in starts] == [
        "Node: node",
        "LLM: model",
        "Tool: xxxxxx",
        "Retriever: y",
        "Tool: safe",
    ]
    assert starts[0][2]["lane_name"] == "node"
    assert starts[0][2]["lane_id"] == "langgraph.no"
    hostile_tool_attributes = cast(dict[str, object], starts[2][2]["attributes"])
    bounded_tool_attributes = cast(dict[str, object], starts[4][2]["attributes"])
    assert "tool_call_id" not in hostile_tool_attributes
    assert bounded_tool_attributes["tool_call_id"] == "z" * 12
    assert all(len(cast(str, call[1][1])) <= 12 for call in starts)


def test_framework_projection_is_cycle_and_depth_safe() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.bounded(max_depth=3),
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)

    handler.on_chain_start(
        {},
        {"cycle": cyclic, "nested": {"value": {"leaf": {"data": "unreached"}}}},
        run_id=UUID(int=1),
        name="chain",
    )

    attributes = cast(dict[str, object], recorder.calls[0][2]["attributes"])
    assert attributes["input"] == {
        "cycle": ["<cycle>"],
        "nested": {"value": {"leaf": "<max-depth>"}},
    }


def test_framework_projection_does_not_fully_traverse_large_mappings() -> None:
    class CountingMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.visited = 0

        def __getitem__(self, key: str) -> object:
            return key

        def __iter__(self) -> Iterator[str]:
            for index in range(100):
                self.visited += 1
                yield f"key-{index}"

        def __len__(self) -> int:
            raise AssertionError("projection must not size arbitrary mappings")

    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.bounded(max_collection_items=3),
    )
    inputs = CountingMapping()

    handler.on_chain_start({}, cast(Any, inputs), run_id=UUID(int=1), name="chain")

    attributes = cast(dict[str, object], recorder.calls[0][2]["attributes"])
    assert inputs.visited == 4
    assert attributes["input"] == {
        "key-0": "key-0",
        "key-1": "key-1",
        "key-2": "key-2",
        "<truncated>": 1,
    }

    metadata_recorder = FakeRecorder()
    metadata_handler = LangGraphCallbackHandler(
        metadata_recorder,
        capture=CapturePolicy.metadata(max_collection_items=3),
    )
    metadata_inputs = CountingMapping()
    metadata_handler.on_chain_start(
        {}, cast(Any, metadata_inputs), run_id=UUID(int=2), name="chain"
    )

    metadata_attributes = cast(dict[str, object], metadata_recorder.calls[0][2]["attributes"])
    assert metadata_inputs.visited == 0
    assert metadata_attributes["input"] == {"type": "mapping"}


def test_chat_and_generation_projection_bounds_framework_objects() -> None:
    recorder = FakeRecorder()
    handler = LangGraphCallbackHandler(
        recorder,
        capture=CapturePolicy.bounded(max_collection_items=3),
    )
    messages: list[BaseMessage] = [HumanMessage(content=f"message-{index}") for index in range(10)]

    handler.on_chat_model_start({}, [messages], run_id=UUID(int=1))

    chat_attributes = cast(dict[str, object], recorder.calls[0][2]["attributes"])
    assert chat_attributes["messages"] == [
        [
            {"type": "human", "content": "message-0"},
            {"type": "human", "content": "message-1"},
            {"type": "human", "content": "message-2"},
            "<truncated:7>",
        ]
    ]

    generations = [[ChatGeneration(message=message) for message in messages]]
    handler.on_llm_end(LLMResult(generations=generations), run_id=UUID(int=1))

    end_attributes = cast(dict[str, object], _call(recorder, "end_span")[1]["attributes"])
    output = cast(dict[str, object], end_attributes["output"])
    assert output["generations"] == [
        [
            {"type": "human", "content": "message-0"},
            {"type": "human", "content": "message-1"},
            {"type": "human", "content": "message-2"},
            "<truncated:7>",
        ]
    ]


@pytest.mark.parametrize("bad_id", ["not-a-uuid", 1])
def test_callback_ids_must_be_uuids(bad_id: object) -> None:
    handler = LangGraphCallbackHandler(FakeRecorder(), raise_error=True)

    with pytest.raises(TypeError, match="run_id must be a UUID"):
        handler.on_chain_end({}, run_id=cast(Any, bad_id))
