"""Real graph -> public session -> artifact -> provisioned Perfetto SQL."""

import asyncio
import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from langgraph_instrumentation import CapturePolicy, RecordingSession, SpanKind


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


def query(binary: str, artifact: Path, sql: str) -> list[dict[str, str]]:
    result = subprocess.run(
        [binary, str(artifact), "-Q", sql],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


@pytest.mark.perfetto
@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_perfetto_import_and_sql(local_agent, asynchronous, tmp_path):
    binary = os.environ.get("PERFETTO_TRACE_PROCESSOR")
    assert binary and Path(binary).is_file(), (
        "Provision pinned tooling with scripts/install_perfetto.py and set PERFETTO_TRACE_PROCESSOR"
    )
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
    bundle = session.snapshot()[0]
    imported = query(
        binary,
        artifact,
        (
            "SELECT dur, extract_arg(arg_set_id,'args.span_id') AS span_id, "
            "extract_arg(arg_set_id,'args.parent_span_id') AS parent_span_id FROM slice"
        ),
    )
    by_id = {row["span_id"]: row for row in imported}
    assert len(by_id) == len(bundle.spans)
    for span in bundle.spans:
        row = by_id[str(span.span_id)]
        assert span.duration_ns is not None
        assert abs(int(row["dur"]) - span.duration_ns) <= 1
        assert row["parent_span_id"] == (
            str(span.parent_span_id) if span.parent_span_id else "[NULL]"
        )
    counters = query(binary, artifact, "SELECT max(value) AS total FROM counter WHERE value = 8")
    assert float(counters[0]["total"]) == 8
    flows = query(binary, artifact, "SELECT count(*) AS count FROM flow")
    assert int(flows[0]["count"]) == len(bundle.spans) - 1
    errors = query(
        binary, artifact, "SELECT name, value FROM stats WHERE severity = 'error' AND value > 0"
    )
    assert not errors
