"""Transactional local trace persistence. No recorder or framework dependencies."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ..identifiers import TraceId
from ..models import (
    MetricPoint,
    Span,
    SpanEvent,
    SpanStatus,
    Trace,
    TraceBundle,
    TraceQuery,
    TraceSummary,
)
from ._validation import _active_trace, _same_started_span, _validate_timeout

_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE traces(id TEXT PRIMARY KEY, start_ns INTEGER NOT NULL, end_ns INTEGER,
                    service TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE spans(trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                   id TEXT NOT NULL, parent_id TEXT, status TEXT NOT NULL, end_ns INTEGER,
                   start_data TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(trace_id,id));
CREATE TABLE events(seq INTEGER PRIMARY KEY, trace_id TEXT NOT NULL, span_id TEXT NOT NULL,
                    data TEXT NOT NULL, FOREIGN KEY(trace_id,span_id)
                    REFERENCES spans(trace_id,id) ON DELETE CASCADE);
CREATE TABLE metrics(seq INTEGER PRIMARY KEY, trace_id TEXT NOT NULL
                     REFERENCES traces(id) ON DELETE CASCADE, data TEXT NOT NULL);
CREATE TABLE diagnostics(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE INDEX trace_list ON traces(start_ns DESC,id DESC);
CREATE INDEX trace_filter ON traces(service,status,start_ns);
CREATE INDEX span_status ON spans(trace_id,status);
CREATE INDEX span_end ON spans(trace_id,end_ns);
CREATE INDEX span_parent ON spans(trace_id,parent_id);
CREATE INDEX event_span ON events(trace_id,span_id,seq);
CREATE INDEX metric_trace ON metrics(trace_id,seq);
PRAGMA user_version=1;
COMMIT;
"""


def _json(record) -> str:
    return json.dumps(record.to_dict(), separators=(",", ":"), allow_nan=False)


class SQLiteTraceStore:
    """WAL, serialized connection ownership, and atomic per-operation transactions.

    ``recover=True`` is an explicit assertion that previous writers have stopped.
    Normal opens do not abandon another process's active work. First schema is v1;
    unknown versions and nonempty unversioned databases are never overwritten.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout: float = 2.0,
        recover: bool = False,
        read_only: bool = False,
    ) -> None:
        _validate_timeout(busy_timeout)
        if read_only and recover:
            raise ValueError("recovery requires writable exclusive ownership")
        self._lock = threading.RLock()
        self._closed = False
        self._read_only = read_only
        target = Path(path).resolve().as_uri() + "?mode=ro" if read_only else str(path)
        self._db = sqlite3.connect(
            target,
            timeout=busy_timeout,
            isolation_level=None,
            check_same_thread=False,
            uri=read_only,
        )
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported trace schema version {version}; expected 1")
            if version == 0:
                if read_only:
                    raise ValueError("read-only database has no recording schema")
                if self._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                    raise ValueError("refusing nonempty unversioned database")
                # Fixed internal DDL only, executed under the initialization lock.
                # Unlike executescript, this does not implicitly commit first.
                for statement in _SCHEMA.split(";"):
                    sql = statement.strip()
                    if sql and sql not in ("BEGIN IMMEDIATE", "COMMIT"):
                        self._db.execute(sql)
            self._db.commit()
            if not read_only:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=FULL")
            if recover:
                self.recover_unfinished()
        except BaseException:
            self._db.close()
            raise

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("trace store is closed")
            if write and self._read_only:
                raise RuntimeError("recording store is read-only")
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @property
    def diagnostics(self) -> dict[str, int]:
        with self._transaction() as db:
            return {r[0]: r[1] for r in db.execute("SELECT key,value FROM diagnostics")}

    def record_diagnostics(self, values: Mapping[str, int]) -> None:
        with self._transaction(write=True) as db:
            for key, value in values.items():
                if (
                    not isinstance(key, str)
                    or len(key) > 128
                    or type(value) is not int
                    or value < 0
                ):
                    raise ValueError("invalid recording diagnostic")
                db.execute(
                    "INSERT INTO diagnostics VALUES (?,?) ON CONFLICT(key) "
                    "DO UPDATE SET value=excluded.value",
                    (f"recording.{key}", value),
                )

    @staticmethod
    def _trace(db, trace_id):
        row = db.execute("SELECT data FROM traces WHERE id=?", (str(trace_id),)).fetchone()
        if row is None:
            raise ValueError("trace does not exist")
        return Trace.from_dict(json.loads(row[0]))

    @staticmethod
    def _span(db, trace_id, span_id):
        row = db.execute(
            "SELECT * FROM spans WHERE trace_id=? AND id=?", (str(trace_id), str(span_id))
        ).fetchone()
        if row is None:
            raise ValueError("span does not exist")
        span = Span.from_dict(json.loads(row["data"]))
        events = tuple(
            SpanEvent.from_dict(json.loads(r[0]))
            for r in db.execute(
                "SELECT data FROM events WHERE trace_id=? AND span_id=? ORDER BY seq",
                (str(trace_id), str(span_id)),
            )
        )
        return replace(span, events=events), Span.from_dict(json.loads(row["start_data"]))

    def start_span(self, trace: Trace, span: Span) -> None:
        with self._transaction(write=True) as db:
            if (
                trace.trace_id != span.trace_id
                or trace.end_time_unix_ns is not None
                or span.is_finished
            ):
                raise ValueError("start_span requires matching active trace and span")
            if span.start_time_unix_ns < trace.start_time_unix_ns:
                raise ValueError("span cannot start before trace")
            exists = db.execute(
                "SELECT id FROM traces WHERE id=?", (str(trace.trace_id),)
            ).fetchone()
            if exists:
                if _active_trace(self._trace(db, trace.trace_id)) != trace:
                    raise ValueError("conflicting trace metadata")
                existing = db.execute(
                    "SELECT start_data FROM spans WHERE trace_id=? AND id=?",
                    (str(trace.trace_id), str(span.span_id)),
                ).fetchone()
                if existing:
                    if Span.from_dict(json.loads(existing[0])) == span:
                        return
                    raise ValueError("span already exists with conflicting start")
            if span.parent_span_id is not None and (
                not exists
                or not db.execute(
                    "SELECT id FROM spans WHERE trace_id=? AND id=?",
                    (str(trace.trace_id), str(span.parent_span_id)),
                ).fetchone()
            ):
                raise ValueError("parent span does not exist")
            if not exists:
                db.execute(
                    "INSERT INTO traces VALUES (?,?,?,?,?,?)",
                    (
                        str(trace.trace_id),
                        trace.start_time_unix_ns,
                        None,
                        trace.service_name,
                        SpanStatus.UNSET.value,
                        _json(trace),
                    ),
                )
            else:
                db.execute(
                    "UPDATE traces SET end_ns=NULL,status=?,data=? WHERE id=?",
                    (SpanStatus.UNSET.value, _json(trace), str(trace.trace_id)),
                )
            db.execute(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?)",
                (
                    str(span.trace_id),
                    str(span.span_id),
                    str(span.parent_span_id) if span.parent_span_id else None,
                    SpanStatus.UNSET.value,
                    None,
                    _json(span),
                    _json(replace(span, events=())),
                ),
            )
            for event in span.events:
                db.execute(
                    "INSERT INTO events(trace_id,span_id,data) VALUES (?,?,?)",
                    (str(event.trace_id), str(event.span_id), _json(event)),
                )

    def record_event(self, event: SpanEvent) -> None:
        with self._transaction(write=True) as db:
            row = db.execute(
                "SELECT data FROM spans WHERE trace_id=? AND id=?",
                (str(event.trace_id), str(event.span_id)),
            ).fetchone()
            if row is None:
                raise ValueError("span does not exist")
            span = Span.from_dict(json.loads(row[0]))
            if span.is_finished or event.time_monotonic_ns < span.start_time_monotonic_ns:
                raise ValueError("event requires active span and valid timestamp")
            db.execute(
                "INSERT INTO events(trace_id,span_id,data) VALUES (?,?,?)",
                (str(event.trace_id), str(event.span_id), _json(event)),
            )

    def record_metric(self, metric: MetricPoint) -> None:
        with self._transaction(write=True) as db:
            self._trace(db, metric.trace_id)
            if (
                metric.span_id is not None
                and not db.execute(
                    "SELECT id FROM spans WHERE trace_id=? AND id=?",
                    (str(metric.trace_id), str(metric.span_id)),
                ).fetchone()
            ):
                raise ValueError("span does not exist")
            db.execute(
                "INSERT INTO metrics(trace_id,data) VALUES (?,?)",
                (str(metric.trace_id), _json(metric)),
            )

    def complete_span(self, span: Span) -> None:
        with self._transaction(write=True) as db:
            self._complete(db, span)

    def _complete(self, db, span):
        if not span.is_finished or span.status is SpanStatus.UNSET:
            raise ValueError("completion requires finished span and terminal status")
        active, started = self._span(db, span.trace_id, span.span_id)
        if active.is_finished:
            if active == span:
                return
            raise ValueError("span already complete with conflicting data")
        if not _same_started_span(started, span) or active.events != span.events:
            raise ValueError("completion conflicts with started span or recorded events")
        db.execute(
            "UPDATE spans SET status=?,end_ns=?,data=? WHERE trace_id=? AND id=?",
            (
                span.status.value,
                span.end_time_unix_ns,
                _json(replace(span, events=())),
                str(span.trace_id),
                str(span.span_id),
            ),
        )
        if not db.execute(
            "SELECT 1 FROM spans WHERE trace_id=? AND status=? LIMIT 1",
            (str(span.trace_id), SpanStatus.UNSET.value),
        ).fetchone():
            trace = self._trace(db, span.trace_id)
            end = db.execute(
                "SELECT max(end_ns) FROM spans WHERE trace_id=?", (str(span.trace_id),)
            ).fetchone()[0]
            statuses = {
                r[0]
                for r in db.execute(
                    "SELECT status FROM spans WHERE trace_id=? AND parent_id IS NULL",
                    (str(span.trace_id),),
                )
            }
            status = next(
                (
                    s
                    for s in (SpanStatus.ERROR, SpanStatus.CANCELLED, SpanStatus.ABANDONED)
                    if s.value in statuses
                ),
                SpanStatus.SUCCESS,
            )
            db.execute(
                "UPDATE traces SET end_ns=?,status=?,data=? WHERE id=?",
                (
                    end,
                    status.value,
                    _json(replace(trace, end_time_unix_ns=end)),
                    str(span.trace_id),
                ),
            )

    def get_trace(self, trace_id: TraceId) -> TraceBundle | None:
        with self._transaction() as db:
            if not db.execute("SELECT id FROM traces WHERE id=?", (str(trace_id),)).fetchone():
                return None
            events = defaultdict(list)
            for row in db.execute(
                "SELECT data FROM events WHERE trace_id=? ORDER BY seq", (str(trace_id),)
            ):
                event = SpanEvent.from_dict(json.loads(row[0]))
                events[event.span_id].append(event)
            spans = []
            for row in db.execute("SELECT data FROM spans WHERE trace_id=?", (str(trace_id),)):
                span = Span.from_dict(json.loads(row[0]))
                spans.append(replace(span, events=tuple(events[span.span_id])))
            metrics = tuple(
                MetricPoint.from_dict(json.loads(r[0]))
                for r in db.execute(
                    "SELECT data FROM metrics WHERE trace_id=? ORDER BY seq", (str(trace_id),)
                )
            )
            return TraceBundle(self._trace(db, trace_id), tuple(spans), metrics)

    def list_traces(self, query: TraceQuery | None = None) -> tuple[TraceSummary, ...]:
        with self._transaction() as db:
            query = query or TraceQuery()
            clauses, values = [], []
            for column, operator, value in (
                ("t.start_ns", ">=", query.start_time_unix_ns),
                ("t.start_ns", "<=", query.end_time_unix_ns),
                ("t.service", "=", query.service_name),
                ("t.status", "=", query.status.value if query.status else None),
            ):
                if value is not None:
                    clauses.append(f"{column}{operator}?")
                    values.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = db.execute(
                "SELECT t.*, (SELECT count(*) FROM spans s WHERE s.trace_id=t.id) AS count "
                "FROM traces t" + where + " ORDER BY t.start_ns DESC,t.id DESC LIMIT ? OFFSET ?",
                (*values, query.limit, query.offset),
            )
            return tuple(
                TraceSummary(
                    TraceId.from_hex(r["id"]),
                    json.loads(r["data"])["name"],
                    r["service"],
                    r["start_ns"],
                    r["end_ns"],
                    r["count"],
                    SpanStatus(r["status"]),
                )
                for r in rows
            )

    def delete_trace(self, trace_id: TraceId) -> bool:
        with self._transaction(write=True) as db:
            return db.execute("DELETE FROM traces WHERE id=?", (str(trace_id),)).rowcount > 0

    def delete_traces(self, *, before_unix_ns: int) -> int:
        with self._transaction(write=True) as db:
            if type(before_unix_ns) is not int or before_unix_ns < 0:
                raise ValueError("cutoff must be nonnegative integer nanoseconds")
            return db.execute("DELETE FROM traces WHERE end_ns < ?", (before_unix_ns,)).rowcount

    def recover_unfinished(self) -> int:
        with self._transaction(write=True) as db:
            rows = db.execute(
                "SELECT trace_id,id FROM spans WHERE status=?", (SpanStatus.UNSET.value,)
            ).fetchall()
            for row in rows:
                span, _ = self._span(db, row[0], row[1])
                last = max(
                    (e.time_monotonic_ns for e in span.events), default=span.start_time_monotonic_ns
                )
                self._complete(
                    db,
                    replace(
                        span,
                        end_time_monotonic_ns=last,
                        end_time_unix_ns=span.start_time_unix_ns
                        + last
                        - span.start_time_monotonic_ns,
                        status=SpanStatus.ABANDONED,
                        status_description="recovered unfinished span",
                        attributes={
                            **span.attributes,
                            "recording.recovered": True,
                            "duration.lower_bound": True,
                        },
                    ),
                )
            if rows:
                db.execute(
                    "INSERT INTO diagnostics VALUES ('recovered_spans',?) ON CONFLICT(key) "
                    "DO UPDATE SET value=value+excluded.value",
                    (len(rows),),
                )
            return len(rows)

    def close(self, timeout: float | None = None) -> bool:
        _validate_timeout(timeout)
        acquired = self._lock.acquire() if timeout is None else self._lock.acquire(timeout=timeout)
        if not acquired:
            return False
        try:
            if not self._closed:
                self._db.close()
                self._closed = True
            return True
        finally:
            self._lock.release()
