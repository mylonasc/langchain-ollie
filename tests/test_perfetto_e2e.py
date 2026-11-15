"""Real graph -> public session -> artifact -> provisioned Perfetto SQL."""

import asyncio
import json
from typing import Any, cast

import pytest
from perfetto_support import assert_imported_spans, query

from langchain_ollie import CapturePolicy, RecordingSession, SpanKind


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_tool_calling_agent_exports_after_close(local_agent, asynchronous, tmp_path):
    with RecordingSession(capture=CapturePolicy.metadata(), strict=True) as session:
        inputs = {"messages": [("user", "SECRET-CANARY")]}
        config = {"callbacks": [session.handler]}
        if asynchronous:
            result = asyncio.run(local_agent.ainvoke(inputs, config=config))
        else:
            result = local_agent.invoke(inputs, config=config)
    assert result["messages"][-1].content == "Result: 5"
    bundle = session.snapshot()[0]
    assert len([s for s in bundle.spans if s.kind is SpanKind.LLM]) == 2
    assert any(s.name == "Tool: add" for s in bundle.spans)
    assert (
        sum(
            cast(Any, s.attributes["usage"])["total_tokens"]
            for s in bundle.spans
            if s.kind is SpanKind.LLM
        )
        == 8
    )
    assert not any(session.recorder.diagnostics.values())
    destination = tmp_path / "agent.json"
    session.export(destination)
    artifact = json.loads(destination.read_text())
    assert "SECRET-CANARY" not in destination.read_text()
    assert not artifact["otherData"]["incomplete"]
    assert len([e for e in artifact["traceEvents"] if e["ph"] == "X"]) == len(bundle.spans)


@pytest.mark.perfetto
@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_perfetto_import_and_sql(local_agent, asynchronous, tmp_path, trace_processor):
    binary = trace_processor
    with RecordingSession(strict=True) as session:
        config = {"callbacks": [session.handler]}
        inputs = {"messages": [("user", "add")]}
        if asynchronous:
            asyncio.run(local_agent.ainvoke(inputs, config=config))
        else:
            local_agent.invoke(inputs, config=config)
    artifact = tmp_path / "agent.json"
    session.export(artifact)
    rows = query(binary, artifact, "SELECT name, dur FROM slice WHERE dur > 0 ORDER BY name")
    assert any(row["name"] == "Tool: add" for row in rows)
    assert len([r for r in rows if r["name"] == "LLM: ScriptedModel"]) == 2
    assert any(row["name"] == "Graph: LocalAgent" for row in rows)
    assert_imported_spans(binary, artifact, session.snapshot())
    counters = query(binary, artifact, "SELECT max(value) AS total FROM counter WHERE value = 8")
    assert float(counters[0]["total"]) == 8
