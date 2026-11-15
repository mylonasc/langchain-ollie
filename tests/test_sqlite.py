import asyncio
import io
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from perfetto_support import assert_imported_spans

from langchain_ollie import (
    MetricPoint,
    RecordingSession,
    Span,
    SpanEvent,
    SpanId,
    SpanKind,
    SpanLink,
    SpanStatus,
    SQLiteTraceStore,
    Trace,
    TraceId,
    TraceQuery,
)


def add_trace(store, number):
    trace = Trace(TraceId(number), "root", "test", number)
    span = Span(trace.trace_id, SpanId(number), "root", SpanKind.GRAPH, number, number)
    store.start_span(trace, span)
    complete = replace(
        span,
        end_time_unix_ns=number + 1,
        end_time_monotonic_ns=number + 1,
        status=SpanStatus.SUCCESS,
    )
    store.complete_span(complete)
    return trace, span, complete


def test_reopen_pagination_idempotence_and_retention(tmp_path):
    path = tmp_path / "traces.db"
    store = SQLiteTraceStore(path)
    for i in range(1, 5):
        trace, start, end = add_trace(store, i)
        store.start_span(trace, start)
        store.complete_span(end)
    assert [s.trace_id for s in store.list_traces(TraceQuery(limit=1, offset=1))] == [TraceId(3)]
    before = store.get_trace(TraceId(4))
    store.record_diagnostics({"events": 7})
    store.close()
    store = SQLiteTraceStore(path)
    assert store.get_trace(TraceId(4)) == before
    assert store.diagnostics["recording.events"] == 7
    assert store.delete_traces(before_unix_ns=4) == 2
    assert store.delete_trace(TraceId(4))
    with sqlite3.connect(path) as db:
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
    store.close()


def test_transaction_rollback_and_disk_full(tmp_path):
    store = SQLiteTraceStore(tmp_path / "rollback.db")
    trace = Trace(TraceId(1), "root", "test", 100)
    span = Span(trace.trace_id, SpanId(1), "root", SpanKind.GRAPH, 100, 100)
    store.start_span(trace, span)
    before = store.get_trace(trace.trace_id)
    store._db.execute(
        "CREATE TRIGGER fail_update BEFORE UPDATE ON spans "
        "BEGIN SELECT RAISE(ABORT,'injected'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.complete_span(
            replace(
                span, end_time_unix_ns=200, end_time_monotonic_ns=200, status=SpanStatus.SUCCESS
            )
        )
    assert store.get_trace(trace.trace_id) == before
    store._db.execute("DROP TRIGGER fail_update")
    pages = store._db.execute("PRAGMA page_count").fetchone()[0]
    store._db.execute(f"PRAGMA max_page_count={pages}")
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.record_event(
            SpanEvent(trace.trace_id, span.span_id, "large", 101, 101, {"body": "x" * 1_000_000})
        )
    assert store.get_trace(trace.trace_id) == before
    store.close()


def test_unversioned_database_is_not_overwritten(tmp_path):
    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE user_data(value TEXT)")
    with pytest.raises(ValueError, match="unversioned"):
        SQLiteTraceStore(path)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema version 999"):
        SQLiteTraceStore(path)


def test_read_only_inspection_never_mutates_recording(tmp_path):
    path = tmp_path / "read-only.db"
    store = SQLiteTraceStore(path)
    add_trace(store, 1)
    store.close()
    store = SQLiteTraceStore(path, read_only=True)
    assert store.get_trace(TraceId(1))
    with pytest.raises(RuntimeError, match="read-only"):
        store.delete_trace(TraceId(1))
    store.close()


def test_two_connections_write_and_read_consistent_traces(tmp_path):
    path = tmp_path / "parallel.db"
    a, b = SQLiteTraceStore(path), SQLiteTraceStore(path)

    def write(i):
        add_trace(a if i % 2 else b, i)
        return (a if i % 2 else b).get_trace(TraceId(i))

    with ThreadPoolExecutor(max_workers=4) as pool:
        bundles = list(pool.map(write, range(1, 41)))
    assert all(bundle and bundle.trace.end_time_unix_ns is not None for bundle in bundles)
    assert len(a.list_traces()) == len(b.list_traces()) == 40
    a.close()
    b.close()


def test_crashed_process_recovery_is_explicit_and_lower_bound(tmp_path):
    path = tmp_path / "crash.db"
    code = """
import os,sys
from langchain_ollie import SQLiteTraceStore,TraceRecorder,SpanKind
store=SQLiteTraceStore(sys.argv[1])
recorder=TraceRecorder(store)
span=recorder.start_span('root','root',SpanKind.GRAPH)
recorder.add_event('root','last observed')
print(str(span.trace_id),flush=True)
os._exit(23)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=10
    )
    assert child.returncode == 23
    trace_id = TraceId.from_hex(child.stdout.strip())
    store = SQLiteTraceStore(path)
    unfinished = store.get_trace(trace_id)
    assert unfinished is not None and not unfinished.spans[0].is_finished
    store.close()
    store = SQLiteTraceStore(path, recover=True)
    bundle = store.get_trace(trace_id)
    assert bundle and bundle.spans[0].status is SpanStatus.ABANDONED
    assert bundle.spans[0].attributes["duration.lower_bound"]
    assert bundle.spans[0].events[0].name == "last observed"
    assert store.diagnostics["recovered_spans"] == 1
    assert store.recover_unfinished() == 0
    store.close()


def semantic(bundle):
    names = {s.span_id: s.name for s in bundle.spans}
    return Counter((s.name, s.kind, s.status, names.get(s.parent_span_id)) for s in bundle.spans)


@pytest.mark.parametrize("agent_kind", ["graph", "deepagent"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_agents_survive_sqlite_reopen(
    local_agent, deep_agent_example, agent_kind, tmp_path, asynchronous
):
    path = tmp_path / "agent.db"
    store = SQLiteTraceStore(path)
    if agent_kind == "deepagent":
        app = deep_agent_example["build_deep_agent"]().agent
    else:
        app = local_agent
    with RecordingSession(store=store, strict=True) as session:
        inputs = {"messages": [("user", "SQLITE-PRIVATE-CANARY")]}
        cfg = {"callbacks": [session.handler]}
        result = (
            asyncio.run(app.ainvoke(inputs, config=cfg))
            if asynchronous
            else app.invoke(inputs, config=cfg)
        )
    assert result["messages"][-1].content.startswith("Result:")
    expected = session.snapshot()[0]
    store.close()
    reopened = SQLiteTraceStore(path)
    restored = reopened.get_trace(expected.trace.trace_id)
    assert restored == expected
    assert "SQLITE-PRIVATE-CANARY" not in json.dumps(restored.to_dict())
    from langchain_ollie import PerfettoExporter

    first, second = io.StringIO(), io.StringIO()
    PerfettoExporter().export([expected], first)
    PerfettoExporter().export([restored], second)
    assert first.getvalue() == second.getvalue()
    # Compare semantic records from a separate actual memory-backed run.
    other = (
        deep_agent_example["build_deep_agent"]().agent if agent_kind == "deepagent" else local_agent
    )
    if agent_kind == "graph":
        import runpy
        from pathlib import Path

        other = runpy.run_path(str(Path(__file__).parents[1] / "examples/local_agent.py"))[
            "build_agent"
        ]()
    with RecordingSession() as memory:
        other.invoke({"messages": [("user", "run")]}, config={"callbacks": [memory.handler]})
    assert semantic(restored) == semantic(memory.snapshot()[0])
    reopened.close()


def test_local_cli_lists_summarizes_and_exports(tmp_path):
    path = tmp_path / "cli.db"
    store = SQLiteTraceStore(path)
    trace, _, _ = add_trace(store, 1)
    store.close()

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "langchain_ollie", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

    assert json.loads(cli("list", path, "--limit", 1).stdout)[0]["trace_id"] == str(trace.trace_id)
    assert json.loads(cli("summary", path, trace.trace_id).stdout)["spans"] == 1
    destination = tmp_path / "trace.json"
    cli("export", path, trace.trace_id, destination)
    assert json.loads(destination.read_text())["traceEvents"]


@pytest.mark.perfetto
@pytest.mark.parametrize("agent_kind", ["graph", "deepagent"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_reopened_sqlite_artifact_is_importable(
    local_agent, deep_agent_example, agent_kind, tmp_path, trace_processor, asynchronous
):
    path = tmp_path / "import.db"
    store = SQLiteTraceStore(path)
    app = local_agent if agent_kind == "graph" else deep_agent_example["build_deep_agent"]().agent
    with RecordingSession(store=store, strict=True) as session:
        inputs = {"messages": [("user", "run")]}
        cfg = {"callbacks": [session.handler]}
        if asynchronous:
            asyncio.run(app.ainvoke(inputs, config=cfg))
        else:
            app.invoke(inputs, config=cfg)
    store.close()
    store = SQLiteTraceStore(path)
    bundle = store.get_trace(session.snapshot()[0].trace.trace_id)
    assert bundle is not None
    from langchain_ollie import PerfettoExporter

    destination = tmp_path / "trace.json"
    PerfettoExporter().export([bundle], destination, diagnostics=store.diagnostics)
    assert_imported_spans(trace_processor, destination, [bundle])
    store.close()


def test_busy_writer_failure_leaves_no_partial_trace(tmp_path):
    path = tmp_path / "locked.db"
    store = SQLiteTraceStore(path, busy_timeout=0.01)
    with sqlite3.connect(path) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            add_trace(store, 1)
        blocker.rollback()
    assert store.list_traces() == ()
    add_trace(store, 1)
    assert len(store.list_traces()) == 1
    store.close()


def test_links_events_metrics_and_resource_identity_roundtrip_and_cascade(tmp_path):
    path = tmp_path / "links.db"
    store = SQLiteTraceStore(path)
    trace, _, completed = add_trace(store, 1)
    linked = Span(
        trace.trace_id,
        SpanId(2),
        "linked",
        SpanKind.NODE,
        2,
        2,
        parent_span_id=completed.span_id,
        links=(SpanLink(trace.trace_id, completed.span_id),),
        resource_attributes={"service.name": "synthetic"},
    )
    store.start_span(trace, linked)
    event = SpanEvent(trace.trace_id, linked.span_id, "observed", 3, 3)
    store.record_event(event)
    store.record_metric(
        MetricPoint(trace.trace_id, "usage", 3, 3, {"count": 1}, span_id=linked.span_id)
    )
    store.complete_span(
        replace(
            linked,
            events=(event,),
            end_time_unix_ns=4,
            end_time_monotonic_ns=4,
            status=SpanStatus.SUCCESS,
        )
    )
    before = store.get_trace(trace.trace_id)
    store.close()
    store = SQLiteTraceStore(path)
    assert store.get_trace(trace.trace_id) == before
    assert store.delete_trace(trace.trace_id)
    for table in ("spans", "events", "metrics"):
        assert store._db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    store.close()


@pytest.mark.parametrize("strict", [False, True])
def test_diagnostic_write_failure_is_visible_and_strict_retryable(tmp_path, strict):
    class FailingDiagnostics(SQLiteTraceStore):
        fail = True

        def record_diagnostics(self, values):
            if self.fail:
                raise OSError("private diagnostic failure")
            super().record_diagnostics(values)

    store = FailingDiagnostics(tmp_path / "diagnostics.db")
    session = RecordingSession(store=store, strict=strict)
    session.recorder.start_span("root", "root", SpanKind.GRAPH)
    session.recorder.end_span("root")
    if strict:
        with pytest.raises(OSError):
            session.close()
        store.fail = False
        assert session.close()
        assert store.diagnostics["recording.destination_failures"] == 1
    else:
        assert not session.close()
        assert not session.close()
    assert session.diagnostics["destination_failures"] == 1
    store.close()
