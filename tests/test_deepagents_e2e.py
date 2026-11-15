"""Real Deep Agents, including delegation and streamed tool-call fragments."""

import asyncio
import io
import json
from collections import Counter

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from perfetto_support import assert_imported_spans, query

from langchain_ollie import (
    CaptureMode,
    CapturePolicy,
    RecordingSession,
    SpanKind,
    SpanStatus,
)

PRIVATE = "PRIVATE-DEEPAGENT-CANARY"
VISIBLE = "PUBLIC-REPORT-PREFIX"
REPORT = VISIBLE + "x" * 200 + PRIVATE
MODES = ("invoke", "ainvoke", "stream", "astream")


def run_agent(scenario, session, mode):
    inputs = {"messages": [HumanMessage(content=REPORT, id="same-input-id")]}
    config = {"callbacks": [session.handler]}
    if mode == "invoke":
        return scenario.agent.invoke(inputs, config=config), []
    if mode == "ainvoke":
        return asyncio.run(scenario.agent.ainvoke(inputs, config=config)), []

    def consume(events):
        final = None
        chunks = []
        for namespace, kind, value in events:
            if kind == "values" and not namespace:
                final = value
            if kind == "messages":
                message, _metadata = value
                if isinstance(message, AIMessageChunk):
                    chunks.append(message)
        assert final is not None
        return final, chunks

    if mode == "stream":
        return consume(
            scenario.agent.stream(
                inputs,
                config=config,
                stream_mode=["messages", "values"],
                subgraphs=True,
            )
        )

    async def collect():
        events = [
            event
            async for event in scenario.agent.astream(
                inputs,
                config=config,
                stream_mode=["messages", "values"],
                subgraphs=True,
            )
        ]
        return consume(events)

    return asyncio.run(collect())


def assert_execution(example, scenario, result, chunks, mode):
    assert result["messages"][-1].content == example["FINAL_RESPONSE"]
    assert result["files"][example["FILE_PATH"]]["content"] == REPORT
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert {message.name for message in tool_messages} == {"write_file", "read_file", "task"}
    assert all(message.status != "error" for message in tool_messages)
    assert {message.name: message.tool_call_id for message in tool_messages} == {
        "write_file": "write",
        "read_file": "read",
        "task": "delegate",
    }
    assert "worker-result: 10" in next(m.content for m in tool_messages if m.name == "task")
    assert scenario.coordinator.bound_tools == {"write_file", "read_file", "task"}
    assert scenario.worker.bound_tools == {"read_file", "multiply"}
    if mode in ("stream", "astream"):
        assert any(chunk.tool_call_chunks for chunk in chunks)
        assert any(chunk.content for chunk in chunks)
        # The stream also exposes nested worker chunks; usage appears once/call.
        assert sum((chunk.usage_metadata or {}).get("total_tokens", 0) for chunk in chunks) == 18


def assert_semantics(bundle):
    roots = [span for span in bundle.spans if span.parent_span_id is None]
    assert len(roots) == 1 and roots[0].kind is SpanKind.GRAPH
    assert roots[0].name == "Graph: OfflineDeepAgent"
    assert all(span.status is SpanStatus.SUCCESS for span in bundle.spans)
    models = [span for span in bundle.spans if span.kind is SpanKind.LLM]
    assert Counter(span.name for span in models) == {"LLM: coordinator": 4, "LLM: worker": 2}
    assert sum(span.attributes["usage"]["total_tokens"] for span in models) == 18
    assert len(bundle.metrics) == 6
    assert max(metric.values["total_tokens"] for metric in bundle.metrics) == 18
    assert {span.name for span in bundle.spans if span.kind is SpanKind.TOOL} == {
        "Tool: write_file",
        "Tool: read_file",
        "Tool: task",
        "Tool: multiply",
    }
    by_id = {span.span_id: span for span in bundle.spans}
    delegated = [span for span in bundle.spans if span.name in ("LLM: worker", "Tool: multiply")]
    for span in delegated:
        ancestors = []
        while span.parent_span_id is not None:
            span = by_id[span.parent_span_id]
            ancestors.append(span.name)
        assert "Tool: task" in ancestors


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "capture",
    [
        CapturePolicy.none(),
        CapturePolicy.metadata(),
        CapturePolicy.bounded(max_string_length=80),
    ],
    ids=["none", "metadata", "bounded"],
)
def test_real_deep_agent_privacy_and_usage(deep_agent_example, mode, capture, caplog, tmp_path):
    scenario = deep_agent_example["build_deep_agent"](report=REPORT)
    with RecordingSession(capture=capture, strict=True) as session:
        result, chunks = run_agent(scenario, session, mode)
    assert_execution(deep_agent_example, scenario, result, chunks, mode)
    assert scenario.coordinator.calls == 4 and scenario.worker.calls == 2
    assert len(session.snapshot()) == 1
    bundle = session.snapshot()[0]
    assert_semantics(bundle)
    serialized = json.dumps(bundle.to_dict())
    destination = tmp_path / "deep-agent.json"
    session.export(destination)
    assert PRIVATE not in serialized and PRIVATE not in destination.read_text()
    assert PRIVATE not in caplog.text
    assert all(not span.events for span in bundle.spans), "per-token capture is opt-in"
    assert not session.diagnostics["destination_failures"]
    assert not session.diagnostics["lifecycle"]
    if capture.mode is CaptureMode.BOUNDED:
        assert VISIBLE in serialized
        assert "truncated" in serialized
        assert json.loads(destination.read_text())["otherData"]["incomplete"]
    else:
        assert VISIBLE not in serialized
        assert not any(session.diagnostics.values())
    assert list(tmp_path.iterdir()) == [destination], "StateBackend must not write host files"


@pytest.mark.parametrize("mode", MODES)
def test_repeated_deep_agent_runs_are_isolated(deep_agent_example, mode):
    scenario = deep_agent_example["build_deep_agent"](report=REPORT, invocations=2)
    with RecordingSession(
        capture=CapturePolicy.bounded(max_string_length=80), strict=True
    ) as session:
        for _ in range(2):
            result, chunks = run_agent(scenario, session, mode)
            assert_execution(deep_agent_example, scenario, result, chunks, mode)
    bundles = session.snapshot()
    assert len(bundles) == 2
    assert bundles[0].trace.trace_id != bundles[1].trace.trace_id
    assert scenario.coordinator.calls == 8 and scenario.worker.calls == 4
    for bundle in bundles:
        assert_semantics(bundle)
        serialized = json.dumps(bundle.to_dict())
        assert VISIBLE in serialized and PRIVATE not in serialized
        assert "repeated" not in serialized
    output = io.StringIO()
    session.export(output)
    artifact = json.loads(output.getvalue())
    assert len(artifact["otherData"]["resources"]) == 2


def test_scripted_model_fails_on_extra_calls_and_returns_copies(deep_agent_example):
    response = AIMessage(content="scripted response")
    model = deep_agent_example["ScriptedModel"](responses=(response, response))
    first = model.invoke("run")
    first.content = "caller mutation"
    assert model.invoke("run").content == "scripted response"
    assert response.content == "scripted response"
    with pytest.raises(AssertionError, match="extra scripted model call"):
        model.invoke("unexpected call")
    assert model.calls == 2


def test_scripted_model_rejects_unbound_tools_and_wrong_results(deep_agent_example):
    model = deep_agent_example["ScriptedModel"](
        responses=(
            deep_agent_example["_reply"](
                name="multiply",
                args={"a": 2, "b": 5},
                call_id="test",
            ),
        ),
    )
    with pytest.raises(AssertionError, match="runtime did not bind"):
        model.invoke("run")
    assert model.calls == 0
    model.bind_tools([deep_agent_example["multiply"]])
    assert model.invoke("run").tool_calls[0]["id"] == "test"
    checked = deep_agent_example["ScriptedModel"](
        responses=(AIMessage(content="done"),),
        expectations=(("multiply", "10"),),
    )
    with pytest.raises(AssertionError, match="expected tool result"):
        checked.invoke("not a tool result")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_scripted_stream_preserves_tool_arguments_and_single_usage(
    deep_agent_example, asynchronous
):
    model = deep_agent_example["ScriptedModel"](
        responses=(
            deep_agent_example["_reply"](
                name="multiply",
                args={"a": 2, "b": 5},
                call_id="fragmented-call",
            ),
        )
    )
    model.bind_tools([deep_agent_example["multiply"]])

    async def collect():
        return [chunk async for chunk in model.astream("run")]

    chunks = asyncio.run(collect()) if asynchronous else list(model.stream("run"))
    assert sum(bool(chunk.usage_metadata) for chunk in chunks) == 1
    assert sum((chunk.usage_metadata or {}).get("total_tokens", 0) for chunk in chunks) == 3
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined = combined + chunk
    assert combined.tool_calls == [
        {
            "name": "multiply",
            "args": {"a": 2, "b": 5},
            "id": "fragmented-call",
            "type": "tool_call",
        }
    ]
    assert model.calls == 1


@pytest.mark.perfetto
@pytest.mark.parametrize("mode", MODES)
def test_real_deep_agent_perfetto_import(deep_agent_example, mode, trace_processor, tmp_path):
    scenario = deep_agent_example["build_deep_agent"](report=REPORT)
    with RecordingSession(strict=True) as session:
        result, chunks = run_agent(scenario, session, mode)
    assert_execution(deep_agent_example, scenario, result, chunks, mode)
    destination = tmp_path / "deep-agent.json"
    session.export(destination)
    assert PRIVATE not in destination.read_text()
    assert_imported_spans(trace_processor, destination, session.snapshot())
    models = query(
        trace_processor,
        destination,
        ("SELECT name, count(*) AS count FROM slice WHERE name LIKE 'LLM:%' GROUP BY name"),
    )
    assert {row["name"]: int(row["count"]) for row in models} == {
        "LLM: coordinator": 4,
        "LLM: worker": 2,
    }
    counters = query(trace_processor, destination, "SELECT max(value) AS total FROM counter")
    assert float(counters[0]["total"]) == 18
