"""Thread-safe trace span lifecycle coordination."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import islice
from typing import Protocol, Self, TypeVar, cast

from .clock import Clock, SystemClock
from .identifiers import IdGenerator, RandomIdGenerator, SpanId, TraceId
from .limits import RecordingLimits
from .models import (
    Attributes,
    ExecutionLane,
    MetricPoint,
    Span,
    SpanEvent,
    SpanKind,
    SpanLink,
    SpanStatus,
    Trace,
    freeze_attributes,
)
from .policies import RedactionPolicy, Sanitizer
from .stores.base import FlushableTraceStore, TraceStore

_LOGGER = logging.getLogger(__name__)
_Result = TypeVar("_Result")


class SpanProcessor(Protocol):
    """Consumes completed immutable spans using cooperative timeout budgets.

    Processor callbacks must not synchronously call ``force_flush`` or ``close``
    on their recorder. The recorder rejects such re-entry rather than waiting
    for the callback that is making the call.
    """

    def on_end(self, span: Span) -> None: ...

    def force_flush(self, timeout: float | None = None) -> bool: ...

    def shutdown(self, timeout: float | None = None) -> bool: ...


class RecorderError(RuntimeError):
    """Raised for lifecycle or instrumentation failures in strict mode."""


class _ClosingError(RuntimeError):
    pass


class _AdmissionError(RecorderError):
    pass


class _RetryableCloseError(RecorderError):
    pass


@dataclass(slots=True)
class _ActiveSpan:
    correlation_id: str
    trace: Trace
    span: Span
    events: list[SpanEvent]
    chunks: int = 0
    first_chunk_ns: int | None = None
    last_chunk_ns: int | None = None
    persisted: bool = True


@dataclass(frozen=True, slots=True)
class _SpanContext:
    trace: Trace
    span_id: SpanId
    execution_lane: ExecutionLane | None
    status: SpanStatus
    persisted: bool = True

    @property
    def trace_id(self) -> TraceId:
        return self.trace.trace_id


@dataclass(slots=True)
class _Operation:
    callback: Callable[[], object]
    done: bool = False
    cancelled: bool = False
    result: object = None
    error: BaseException | None = None


@dataclass(slots=True)
class _CloseAttempt:
    done: threading.Event
    result: bool = True
    error: BaseException | None = None
    retryable: bool = False


class TraceRecorder:
    """Coordinate span lifecycle independently of callback execution context.

    Store operations are globally ordered by recorder-call admission. Strict
    recording mode keeps rejected transitions retryable and raises RecorderError.
    Default best-effort mode accepts lifecycle independently of persistence:
    failed destination writes disclose loss, never erase parent context, and never
    block independent processors. Resource-bearing completed spans are dispatched
    once, outside state locks; this is not an exactly-once delivery guarantee.

    ``force_flush`` and ``close`` accept seconds-based caller wait timeouts.
    Shutdown dependencies receive the recorder-wide ``shutdown_timeout`` budget
    configured at construction, so one impatient close caller cannot shorten
    shutdown for every caller. A dependency that ignores its cooperative budget
    can delay background shutdown; hard cancellation is deliberately avoided.
    Completed correlation contexts are retained without payloads in a bounded,
    completion-ordered cache so delayed children can reopen inactive traces.
    In strict mode, rejected abandoned spans remain retryable with dependencies
    open. Best-effort shutdown dispatches accepted spans once and closes resources
    even if persistence failed, returning a stable failure result and diagnostics.
    """

    def __init__(
        self,
        store: TraceStore | None = None,
        *,
        processors: Iterable[SpanProcessor] = (),
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        strict: bool = False,
        shutdown_timeout: float | None = 30.0,
        completed_context_cache_size: int = 4096,
        service_name: str = "langchain-ollie",
        resource_attributes: Attributes | None = None,
        limits: RecordingLimits | None = None,
        owns_store: bool = True,
    ) -> None:
        _validate_timeout(shutdown_timeout, "shutdown_timeout")
        _validate_cache_size(completed_context_cache_size)
        self._store = store
        self._owns_store = owns_store
        self._processors = tuple(processors)
        self._clock = clock or SystemClock()
        self._id_generator = id_generator or RandomIdGenerator()
        self._strict = strict
        self._shutdown_timeout = shutdown_timeout
        self._completed_context_cache_size = completed_context_cache_size
        self._limits = limits or RecordingLimits()
        self._attribute_sanitizer = Sanitizer(
            max_payload_bytes=self._limits.max_attribute_bytes,
            max_depth=64,
            max_string_length=None,
            max_collection_items=None,
            max_byte_length=None,
            redaction=RedactionPolicy(keys=frozenset()),
        )
        self._counts = {"spans": 0, "events": 0, "metrics": 0}
        self._trace_counts: dict[TraceId, dict[str, int]] = {}
        self._attribute_bytes = 0
        self._dropped = {
            "spans": 0,
            "events": 0,
            "metrics": 0,
            "attributes": 0,
            "lanes": 0,
            "destination_failures": 0,
            "lifecycle": 0,
            "bytes": 0,
            "active_spans": 0,
            "expired_spans": 0,
        }
        _validate_nonempty_string(service_name, "service_name")
        self._service_name = service_name[:256]
        self._resource_attributes = freeze_attributes(self._bounded_attributes(resource_attributes))
        self._condition = threading.Condition(threading.Lock())
        self._flush_lock = threading.Lock()
        self._processor_callback = threading.local()
        self._operations: deque[_Operation] = deque()
        self._dispatching = False
        self._dispatcher_thread_id: int | None = None
        self._active: dict[str, _ActiveSpan] = {}
        self._completed_contexts: OrderedDict[str, _SpanContext] = OrderedDict()
        self._lanes: dict[str, ExecutionLane] = {}
        self._metric_totals: dict[tuple[TraceId, str], dict[str, int | float]] = {}
        self._generation = 0
        self._flushed_generation = -1
        self._processor_sequence = 0
        self._active_processor_calls: set[int] = set()
        self._closing = False
        self._closed = False
        self._close_done = threading.Event()
        self._close_result = True
        self._close_error: BaseException | None = None
        self._close_attempt: _CloseAttempt | None = None

    @property
    def diagnostics(self) -> dict[str, int]:
        """Snapshot of lifetime loss counters; nonzero means an incomplete recording."""
        with self._condition:
            return dict(self._dropped)

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def _bounded_attributes(
        self, attributes: Attributes | None, *, count_loss: bool = True
    ) -> Attributes:
        # Direct callers provide trusted JSON values. This is a safety budget,
        # not a substitute for adapter payload capture/redaction configuration.
        result = self._attribute_sanitizer.sanitize(attributes or {})
        if isinstance(result, dict):
            encoded = json.dumps(result)
            if count_loss and any(
                marker in encoded
                for marker in (
                    "<truncated",
                    "<max-depth>",
                    "<node-budget>",
                    "<payload-budget>",
                    "<integer-budget>",
                )
            ):
                self._dropped["attributes"] += 1
            return result
        if count_loss:
            self._dropped["attributes"] += 1
        return {"recording.truncated": result}

    def _admit(self, kind: str) -> bool:
        limit = getattr(self._limits, f"max_{kind}")
        if self._counts[kind] >= limit:
            self._dropped[kind] += 1
            return False
        return True

    def _trace_admit(self, trace_id: TraceId, kind: str) -> bool:
        counts = self._trace_counts.setdefault(
            trace_id, {"spans": 0, "events": 0, "metrics": 0, "bytes": 0}
        )
        if counts[kind] >= getattr(self._limits, f"max_trace_{kind}"):
            self._dropped[kind] += 1
            return False
        return True

    def _charge_attributes(self, trace_id: TraceId, attributes: Attributes) -> Attributes:
        size = len(json.dumps(dict(attributes), separators=(",", ":"), ensure_ascii=True).encode())
        counts = self._trace_counts[trace_id]
        if (
            self._attribute_bytes + size > self._limits.max_total_attribute_bytes
            or counts["bytes"] + size > self._limits.max_trace_attribute_bytes
        ):
            self._dropped["bytes"] += 1
            return {}  # structural fields remain; the recording's loss summary is authoritative
        self._attribute_bytes += size
        counts["bytes"] += size
        return attributes

    def start_span(
        self,
        correlation_id: str,
        name: str,
        kind: SpanKind,
        *,
        parent_correlation_id: str | None = None,
        attributes: Attributes | None = None,
        lane_id: str | None = None,
        lane_name: str | None = None,
        trace_name: str | None = None,
        links: Iterable[SpanLink] = (),
    ) -> Span | None:
        """Start a root or child span and return its immutable active snapshot."""
        _validate_correlation_id(correlation_id)
        self.expire_spans()
        selected_links = tuple(islice(links, 33))
        if parent_correlation_id is not None:
            _validate_correlation_id(parent_correlation_id, "parent_correlation_id")
            if trace_name is not None:
                raise ValueError("trace metadata may only be supplied for a root span")

        def operation() -> Span | None:
            with self._condition:
                lifecycle_error = None
                if correlation_id in self._active or correlation_id in self._completed_contexts:
                    lifecycle_error = f"duplicate span correlation ID: {correlation_id!r}"
                parent = None
                if lifecycle_error is None and parent_correlation_id is not None:
                    active_parent = self._active.get(parent_correlation_id)
                    parent = (
                        _context_from_span(
                            active_parent.trace, active_parent.span, active_parent.persisted
                        )
                        if active_parent is not None
                        else self._completed_contexts.get(parent_correlation_id)
                    )
                    if parent is None and self._strict:
                        lifecycle_error = (
                            f"unknown parent correlation ID: {parent_correlation_id!r}"
                        )
            if lifecycle_error is not None:
                return self._lifecycle_failure(lifecycle_error)
            with self._condition:
                orphan = parent_correlation_id is not None and parent is None
                if orphan:
                    self._dropped["lifecycle"] += 1
                if not self._admit("spans"):
                    return None
                if len(self._active) >= self._limits.max_active_spans:
                    self._dropped["active_spans"] += 1
                    return None
                reading = self._clock.now()
                trace_id = (
                    parent.trace_id if parent is not None else self._id_generator.new_trace_id()
                )
                trace = (
                    parent.trace
                    if parent is not None
                    else Trace(
                        trace_id=trace_id,
                        name=(trace_name or name)[:256],
                        service_name=self._service_name,
                        start_time_unix_ns=reading.unix_time_ns,
                        resource_attributes=self._resource_attributes,
                    )
                )
                lane = self._resolve_lane(lane_id, lane_name, parent)
                if not self._trace_admit(trace_id, "spans"):
                    return None
                span = Span(
                    trace_id=trace_id,
                    span_id=self._id_generator.new_span_id(),
                    parent_span_id=parent.span_id if parent is not None else None,
                    name=name[:256],
                    kind=kind,
                    start_time_unix_ns=reading.unix_time_ns,
                    start_time_monotonic_ns=reading.monotonic_time_ns,
                    attributes=self._charge_attributes(
                        trace_id, self._bounded_attributes(attributes)
                    ),
                    execution_lane=lane,
                    resource_attributes={
                        **self._resource_attributes,
                        "service.name": self._service_name,
                    },
                    links=tuple(
                        replace(
                            link,
                            attributes=self._charge_attributes(
                                trace_id, self._bounded_attributes(link.attributes)
                            ),
                        )
                        for link in selected_links[:32]
                    ),
                )
                if len(selected_links) > 32:
                    self._dropped["attributes"] += 1
                if orphan:
                    span = replace(
                        span,
                        attributes={
                            **span.attributes,
                            "recording.orphan_parent": parent_correlation_id,
                        },
                    )
            succeeded = (
                self._store_start_span(trace, span)
                if self._store and (parent is None or parent.persisted)
                else self._store is None
            )
            if not succeeded:
                if self._strict and self._trace_counts[trace_id]["spans"] == 0:
                    self._trace_counts.pop(trace_id, None)
                if self._strict:
                    raise RecorderError("failed to start span")
            with self._condition:
                self._lanes.setdefault(lane.lane_id, lane)
                self._active[correlation_id] = _ActiveSpan(
                    correlation_id, trace, span, [], persisted=succeeded
                )
                self._counts["spans"] += 1
                self._trace_counts[trace_id]["spans"] += 1
                self._generation += 1
            return span

        return self._ordered_lifecycle(operation, "start a span")

    def add_event(
        self,
        correlation_id: str,
        name: str,
        *,
        attributes: Attributes | None = None,
    ) -> SpanEvent | None:
        """Append an event to an active span."""
        _validate_correlation_id(correlation_id)

        def operation() -> SpanEvent | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                return self._lifecycle_failure(
                    f"cannot add event to unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                if not self._admit("events"):
                    return None
                if not self._trace_admit(active.span.trace_id, "events"):
                    return None
                if len(active.events) >= self._limits.max_span_events:
                    self._dropped["events"] += 1
                    return None
                reading = self._clock.now()
                event = SpanEvent(
                    trace_id=active.span.trace_id,
                    span_id=active.span.span_id,
                    name=name[:256],
                    time_unix_ns=reading.unix_time_ns,
                    time_monotonic_ns=reading.monotonic_time_ns,
                    attributes=self._charge_attributes(
                        active.span.trace_id, self._bounded_attributes(attributes)
                    ),
                )
            succeeded = (
                self._store_call("record event", self._store.record_event, event)
                if self._store and active.persisted
                else True
            )
            if not succeeded:
                if self._strict:
                    raise RecorderError("failed to record event")
                active.persisted = False
            with self._condition:
                active.events.append(event)
                self._counts["events"] += 1
                self._trace_counts[active.span.trace_id]["events"] += 1
                self._generation += 1
            return event

        return self._ordered_lifecycle(operation, "add an event")

    def record_metric(
        self,
        correlation_id: str,
        name: str,
        values: Mapping[str, int | float],
        *,
        attributes: Attributes | None = None,
        cumulative: bool = False,
    ) -> MetricPoint | None:
        """Record metric values, optionally as synchronized trace-wide totals."""
        _validate_correlation_id(correlation_id)

        def operation() -> MetricPoint | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                return self._lifecycle_failure(
                    f"cannot record metric for unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                if not self._admit("metrics"):
                    return None
                if not self._trace_admit(active.span.trace_id, "metrics"):
                    return None
                reading = self._clock.now()
                metric = MetricPoint(
                    trace_id=active.span.trace_id,
                    span_id=active.span.span_id,
                    name=name[:256],
                    time_unix_ns=reading.unix_time_ns,
                    time_monotonic_ns=reading.monotonic_time_ns,
                    values=self._bounded_metric_values(values),
                    attributes=self._charge_attributes(
                        active.span.trace_id, self._bounded_attributes(attributes)
                    ),
                )
                totals_key: tuple[TraceId, str] | None = None
                updated: dict[str, int | float] | None = None
                if cumulative:
                    totals_key = (active.span.trace_id, metric.name)
                    updated = self._metric_totals.get(totals_key, {}).copy()
                    for value_name, value in metric.values.items():
                        if value_name not in updated and len(updated) >= 64:
                            self._dropped["attributes"] += 1
                            continue
                        total = updated.get(value_name, 0) + value
                        if isinstance(total, float) and not math.isfinite(total):
                            raise ValueError(
                                f"cumulative metric value {value_name!r} must be finite"
                            )
                        updated[value_name] = total
                    metric = replace(metric, values=updated)
            succeeded = (
                self._store_call("record metric", self._store.record_metric, metric)
                if self._store and active.persisted
                else True
            )
            if not succeeded and self._strict:
                raise RecorderError("failed to record metric")
            with self._condition:
                if totals_key is not None and updated is not None:
                    self._metric_totals[totals_key] = updated
                self._counts["metrics"] += 1
                self._trace_counts[active.span.trace_id]["metrics"] += 1
                self._generation += 1
            return metric

        return self._ordered_lifecycle(operation, "record a metric")

    def end_span(
        self,
        correlation_id: str,
        *,
        status: SpanStatus = SpanStatus.SUCCESS,
        status_description: str | None = None,
        attributes: Attributes | None = None,
    ) -> Span | None:
        """Finish a span with a terminal status and dispatch its snapshot."""
        _validate_correlation_id(correlation_id)
        if not isinstance(status, SpanStatus):
            raise TypeError("status must be a SpanStatus")
        if status is SpanStatus.UNSET:
            raise ValueError("end_span status must be terminal")

        def operation() -> tuple[Span, int] | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                context = self._completed_contexts.get(correlation_id)
                if context is not None and not self._strict:
                    return None
                return self._lifecycle_failure(
                    f"cannot end unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                completed = self._complete(active, status, status_description, attributes)
                if completed.attributes.get("execution.interrupted"):
                    for ancestor in self._active.values():
                        if (
                            ancestor.span.trace_id == completed.trace_id
                            and ancestor.span.parent_span_id is None
                        ):
                            ancestor.span = replace(
                                ancestor.span,
                                attributes={
                                    **ancestor.span.attributes,
                                    "execution.interrupted": True,
                                },
                            )
            succeeded = (
                self._store_call("complete span", self._store.complete_span, completed)
                if self._store and active.persisted
                else True
            )
            if not succeeded and self._strict:
                raise RecorderError("failed to complete span")
            with self._condition:
                del self._active[correlation_id]
                self._retain_completed_context(
                    correlation_id, active.trace, completed, active.persisted and succeeded
                )
                self._generation += 1
                token = self._begin_processor_call()
                self._release_trace_if_inactive(completed.trace_id)
            return completed, token

        result = self._ordered_lifecycle(operation, "end a span")
        if result is None:
            return None
        completed, token = result
        try:
            self._dispatch_processors(completed)
        finally:
            self._finish_processor_call(token)
        return completed

    def observe_token(self, correlation_id: str) -> None:
        """Count callback chunks without retaining text or writing per-token records."""
        self.expire_spans()

        def operation() -> None:
            with self._condition:
                active = self._active.get(correlation_id)
                if active is None:
                    return
                now = self._clock.now().monotonic_time_ns
                active.chunks += 1
                if active.first_chunk_ns is None:
                    active.first_chunk_ns = now
                active.last_chunk_ns = now

        self._ordered_lifecycle(operation, "observe a streaming chunk")

    def expire_spans(self) -> int:
        """Cooperative maintenance on admission/snapshot/token callbacks; no worker thread."""

        def operation() -> list[tuple[Span, int]]:
            with self._condition:
                now = self._clock.now().monotonic_time_ns
                expired = sorted(
                    (
                        a
                        for a in self._active.values()
                        if now - a.span.start_time_monotonic_ns >= self._limits.max_span_age_ns
                    ),
                    key=lambda a: self._active_depth(a.span),
                    reverse=True,
                )
            completed = []
            for active in expired:
                span = self._complete(active, SpanStatus.ABANDONED, "span expired", None)
                if (
                    self._store
                    and active.persisted
                    and not self._store_call("expire span", self._store.complete_span, span)
                ):
                    if self._strict:
                        continue
                    active.persisted = False
                with self._condition:
                    del self._active[active.correlation_id]
                    self._retain_completed_context(
                        active.correlation_id, active.trace, span, active.persisted
                    )
                    self._release_trace_if_inactive(span.trace_id)
                    self._dropped["expired_spans"] += 1
                    self._generation += 1
                    completed.append((span, self._begin_processor_call()))
            return completed

        with self._condition:
            if self._closing or self._closed or not self._active:
                return 0
        result = self._ordered_lifecycle(operation, "expire spans") or []
        for span, token in result:
            try:
                self._dispatch_processors(span)
            finally:
                self._finish_processor_call(token)
        return len(result)

    def force_flush(self, timeout: float | None = None) -> bool:
        """Establish a quiescence barrier and cooperatively flush destinations."""
        if self._reject_processor_reentry("force flush"):
            return False
        deadline = _deadline(timeout)
        remaining = _remaining(deadline)
        acquired = (
            self._flush_lock.acquire()
            if remaining is None
            else self._flush_lock.acquire(timeout=remaining)
        )
        if not acquired:
            return self._timeout_failure("force flush")
        try:
            try:
                return self._force_flush(deadline)
            except _ClosingError:
                pass
        finally:
            self._flush_lock.release()
        with self._condition:
            attempt = self._close_attempt
            if attempt is None:
                if self._closed:
                    return self._closed_result()
                return self._timeout_failure("force flush")
        return self._wait_for_close(attempt, deadline, "force flush")

    def _force_flush(self, deadline: float | None) -> bool:
        """Flush while the caller owns the flush/shutdown serialization lock."""

        def store_barrier() -> tuple[bool, int, int, bool]:
            with self._condition:
                generation = self._generation
                processor_barrier = self._processor_sequence
                if self._flushed_generation == generation:
                    return True, processor_barrier, generation, False
            succeeded = self._flush_store(deadline)
            return succeeded, processor_barrier, generation, True

        try:
            store_ok, processor_barrier, generation, needs_flush = self._ordered(
                store_barrier, deadline
            )
        except TimeoutError:
            return self._timeout_failure("force flush")
        if not needs_flush:
            return True
        if not self._wait_for_processors(processor_barrier, deadline):
            return self._timeout_failure("force flush")
        processors_ok = self._flush_processors(deadline)
        succeeded = store_ok and processors_ok
        if succeeded:
            with self._condition:
                self._flushed_generation = max(self._flushed_generation, generation)
        if not succeeded and self._strict:
            raise RecorderError("failed to force flush")
        return succeeded

    def close(self, timeout: float | None = None) -> bool:
        """Abandon active spans child-first, then flush and shut down once."""
        if self._reject_processor_reentry("close"):
            return False
        deadline = _deadline(timeout)
        with self._condition:
            if self._closed:
                return self._closed_result()
            attempt = self._close_attempt
            if attempt is None:
                attempt = _CloseAttempt(threading.Event())
                self._close_attempt = attempt
                self._close_done = attempt.done
                self._close_result = True
                self._close_error = None
                self._closing = True
                threading.Thread(
                    target=self._close_worker,
                    args=(attempt,),
                    name="trace-recorder-close",
                    daemon=True,
                ).start()
        return self._wait_for_close(attempt, deadline, "close")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            _LOGGER.warning("recorder close failed while preserving application exception")
        return False

    def _perform_close(self, deadline: float | None) -> tuple[bool, bool]:
        def abandon() -> tuple[list[Span], int, bool]:
            with self._condition:
                active = sorted(
                    self._active.values(),
                    key=lambda item: self._active_depth(item.span),
                    reverse=True,
                )
            completed: list[Span] = []
            completed_correlations: list[str] = []
            succeeded = True
            for item in active:
                span = self._complete(item, SpanStatus.ABANDONED, "recorder closed", None)
                store_ok = (
                    self._store_call("complete span", self._store.complete_span, span)
                    if self._store and item.persisted
                    else True
                )
                succeeded = store_ok and succeeded
                if store_ok or not self._strict:
                    completed.append(span)
                    completed_correlations.append(item.correlation_id)
            with self._condition:
                for correlation_id in completed_correlations:
                    del self._active[correlation_id]
                self._generation += len(completed)
                barrier = self._processor_sequence
            return completed, barrier, succeeded

        try:
            completed, barrier, stores_ok = self._ordered(abandon, allow_during_closing=True)
        except TimeoutError:
            return self._timeout_failure("close"), False
        if not self._wait_for_processors(barrier, deadline):
            return self._finish_close_after_processor_timeout(deadline), False
        if not self._acquire_flush_lock(deadline):
            return self._close_failure(), False
        try:
            processors_ok = True
            for span in completed:
                try:
                    processors_ok = self._dispatch_processors(span) and processors_ok
                except RecorderError:
                    processors_ok = False
            if not stores_ok and self._strict:
                if self._strict:
                    raise _RetryableCloseError("failed to persist abandoned spans")
                return False, True
            store_flush_ok = self._flush_store(deadline)
            processors_flush_ok = self._flush_processors(deadline)
            flush_ok = store_flush_ok and processors_flush_ok
            shutdown_ok = self._shutdown_dependencies(deadline)
        finally:
            self._flush_lock.release()
        succeeded = stores_ok and processors_ok and flush_ok and shutdown_ok
        if not succeeded and self._strict:
            raise RecorderError("failed to close recorder")
        return succeeded, False

    def _close_worker(self, attempt: _CloseAttempt) -> None:
        try:
            attempt.result, attempt.retryable = self._perform_close(
                _deadline(self._shutdown_timeout)
            )
        except _RetryableCloseError as error:
            attempt.error = error
            attempt.retryable = True
        except BaseException as error:
            attempt.error = error
        finally:
            with self._condition:
                self._close_result = attempt.result
                self._close_error = attempt.error
                if not attempt.retryable:
                    self._active.clear()
                    self._completed_contexts.clear()
                    self._metric_totals.clear()
                    self._lanes.clear()
                    self._trace_counts.clear()
                    self._operations.clear()
                    self._closed = True
                self._closing = False
                self._close_attempt = None
                attempt.done.set()
                self._condition.notify_all()

    def _ordered(
        self,
        callback: Callable[[], _Result],
        deadline: float | None = None,
        *,
        allow_during_closing: bool = False,
        bounded_lifecycle: bool = False,
    ) -> _Result:
        operation = _Operation(cast(Callable[[], object], callback))
        with self._condition:
            if (self._closing or self._closed) and not allow_during_closing:
                raise _ClosingError
            if self._dispatching and self._dispatcher_thread_id == threading.get_ident():
                raise RecorderError("a store callback must not re-enter its recorder")
            if bounded_lifecycle and len(self._operations) >= self._limits.max_pending_operations:
                self._dropped.setdefault("pending_operations", 0)
                self._dropped["pending_operations"] += 1
                raise _AdmissionError("pending lifecycle admission budget exhausted")
            self._operations.append(operation)
            while not operation.done:
                if not self._dispatching and self._operations:
                    current = self._operations.popleft()
                    self._dispatching = True
                    self._dispatcher_thread_id = threading.get_ident()
                else:
                    remaining = _remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        if operation in self._operations:
                            self._operations.remove(operation)
                            operation.cancelled = True
                        raise TimeoutError
                    self._condition.wait(remaining)
                    continue
                self._condition.release()
                try:
                    if not current.cancelled:
                        current.result = current.callback()
                except BaseException as error:
                    current.error = error
                finally:
                    self._condition.acquire()
                    current.done = True
                    self._dispatching = False
                    self._dispatcher_thread_id = None
                    self._condition.notify_all()
            if operation.error is not None:
                raise operation.error
            return cast(_Result, operation.result)

    def _ordered_lifecycle(
        self,
        callback: Callable[[], _Result],
        operation: str,
    ) -> _Result | None:
        try:
            return self._ordered(callback, bounded_lifecycle=True)
        except _ClosingError:
            return self._lifecycle_failure(f"cannot {operation} after recorder close has begun")
        except _AdmissionError:
            if self._strict:
                raise
            return None

    def _resolve_lane(
        self,
        lane_id: str | None,
        lane_name: str | None,
        parent: _SpanContext | None,
    ) -> ExecutionLane:
        if lane_id is None and parent is not None and parent.execution_lane is not None:
            return parent.execution_lane
        resolved_id = lane_id or "main"
        resolved_id = resolved_id[:256]
        lane_name = lane_name[:256] if lane_name else None
        lane = self._lanes.get(resolved_id)
        if lane is None:
            if len(self._lanes) >= self._limits.max_lanes:
                self._dropped["lanes"] += 1
                return next(iter(self._lanes.values()))
            lane = ExecutionLane(resolved_id, lane_name, len(self._lanes))
        elif lane_name is not None and lane.name != lane_name:
            raise ValueError(f"execution lane {resolved_id!r} already has a different name")
        return lane

    def _complete(
        self,
        active: _ActiveSpan,
        status: SpanStatus,
        description: str | None,
        attributes: Attributes | None,
    ) -> Span:
        reading = self._clock.now()
        merged_attributes = dict(active.span.attributes)
        if attributes is not None:
            merged_attributes.update(self._bounded_attributes(attributes, count_loss=False))
        if active.chunks:
            merged_attributes["llm.streaming"] = {
                "chunks": active.chunks,
                "first_chunk_offset_ns": (active.first_chunk_ns or 0)
                - active.span.start_time_monotonic_ns,
                "last_chunk_offset_ns": (active.last_chunk_ns or 0)
                - active.span.start_time_monotonic_ns,
            }
        return replace(
            active.span,
            end_time_unix_ns=max(reading.unix_time_ns, active.span.start_time_unix_ns),
            end_time_monotonic_ns=max(
                reading.monotonic_time_ns, active.span.start_time_monotonic_ns
            ),
            status=status,
            status_description=description[:256] if description else None,
            attributes=self._charge_attributes(
                active.span.trace_id, self._bounded_attributes(merged_attributes)
            ),
            events=tuple(active.events),
        )

    def _active_depth(self, span: Span) -> int:
        parents = {item.span.span_id: item.span.parent_span_id for item in self._active.values()}
        depth = 0
        parent_id = span.parent_span_id
        while parent_id in parents:
            depth += 1
            parent_id = parents[parent_id]
        return depth

    def _release_trace_if_inactive(self, trace_id: TraceId) -> None:
        if any(item.span.trace_id == trace_id for item in self._active.values()):
            return
        self._metric_totals = {
            key: totals for key, totals in self._metric_totals.items() if key[0] != trace_id
        }

    def _retain_completed_context(
        self, correlation_id: str, trace: Trace, span: Span, persisted: bool = True
    ) -> None:
        self._completed_contexts[correlation_id] = _context_from_span(trace, span, persisted)
        while len(self._completed_contexts) > self._completed_context_cache_size:
            self._completed_contexts.popitem(last=False)

    def _begin_processor_call(self) -> int:
        self._processor_sequence += 1
        token = self._processor_sequence
        self._active_processor_calls.add(token)
        return token

    def _finish_processor_call(self, token: int) -> None:
        with self._condition:
            self._active_processor_calls.discard(token)
            self._condition.notify_all()

    def _wait_for_processors(self, barrier: int, deadline: float | None) -> bool:
        with self._condition:
            while any(token <= barrier for token in self._active_processor_calls):
                remaining = _remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _dispatch_processors(self, span: Span) -> bool:
        succeeded = True
        errors: list[Exception] = []
        for processor in self._processors:
            try:
                self._invoke_processor(processor.on_end, span)
            except Exception as error:
                succeeded = False
                errors.append(error)
                self._destination_failure("process completed span")
        if errors and self._strict:
            raise RecorderError("failed to process completed span") from errors[0]
        return succeeded

    def _store_call[Value](
        self,
        operation: str,
        callback: Callable[[Value], None],
        value: Value,
    ) -> bool:
        try:
            callback(value)
            return True
        except Exception:
            self._destination_failure(operation)
            return False

    def _store_start_span(self, trace: Trace, span: Span) -> bool:
        try:
            if self._store is not None:
                self._store.start_span(trace, span)
            return True
        except Exception:
            self._destination_failure("start span")
            return False

    def _flush_store(self, deadline: float | None) -> bool:
        if self._store is None:
            return True
        if not isinstance(self._store, FlushableTraceStore):
            return True
        return self._timed_call("flush store", self._store.force_flush, deadline)

    def _flush_processors(self, deadline: float | None) -> bool:
        succeeded = True
        for processor in self._processors:
            succeeded = (
                self._timed_processor_call("flush processor", processor.force_flush, deadline)
                and succeeded
            )
        return succeeded

    def _shutdown_dependencies(self, deadline: float | None) -> bool:
        succeeded = True
        for processor in self._processors:
            succeeded = (
                self._timed_processor_call("shut down processor", processor.shutdown, deadline)
                and succeeded
            )
        succeeded = self._close_store(deadline) and succeeded
        return succeeded

    def _finish_close_after_processor_timeout(
        self,
        deadline: float | None,
    ) -> bool:
        if self._acquire_flush_lock(deadline):
            try:
                self._close_store(deadline)
            finally:
                self._flush_lock.release()
        if self._strict:
            raise RecorderError("timed out waiting for processor callbacks during close")
        return False

    def _acquire_flush_lock(self, deadline: float | None) -> bool:
        remaining = _remaining(deadline)
        if remaining is None:
            self._flush_lock.acquire()
            return True
        return self._flush_lock.acquire(timeout=remaining)

    def _close_store(self, deadline: float | None) -> bool:
        if self._store is None or not self._owns_store:
            return True
        return self._timed_call("close store", self._store.close, deadline)

    def _close_failure(self) -> bool:
        if self._strict:
            raise RecorderError("failed to close recorder")
        return False

    def _timed_call(
        self,
        operation: str,
        callback: Callable[[float | None], bool],
        deadline: float | None,
    ) -> bool:
        remaining = _remaining(deadline)
        try:
            succeeded = callback(remaining)
            if not succeeded:
                self._destination_failure(operation)
            return succeeded
        except Exception:
            self._destination_failure(operation)
            return False

    def _timed_processor_call(
        self,
        operation: str,
        callback: Callable[[float | None], bool],
        deadline: float | None,
    ) -> bool:
        try:
            succeeded = self._invoke_processor(callback, _remaining(deadline))
            if not succeeded:
                self._destination_failure(operation)
            return succeeded
        except Exception:
            self._destination_failure(operation)
            return False

    def _invoke_processor[Value, Result](
        self,
        callback: Callable[[Value], Result],
        value: Value,
    ) -> Result:
        previous = getattr(self._processor_callback, "active", False)
        self._processor_callback.active = True
        try:
            return callback(value)
        finally:
            self._processor_callback.active = previous

    def _reject_processor_reentry(self, operation: str) -> bool:
        if not getattr(self._processor_callback, "active", False):
            return False
        message = f"processor callbacks must not call recorder {operation}"
        if self._strict:
            raise RecorderError(message)
        _LOGGER.warning(message)
        return True

    def _wait_for_close(
        self,
        attempt: _CloseAttempt,
        deadline: float | None,
        operation: str,
    ) -> bool:
        if not attempt.done.wait(_remaining(deadline)):
            return self._timeout_failure(operation)
        if attempt.error is not None:
            raise attempt.error
        return attempt.result

    def _closed_result(self) -> bool:
        if self._close_error is not None:
            raise self._close_error
        return self._close_result

    def _timeout_failure(self, operation: str) -> bool:
        if self._strict:
            raise RecorderError(f"timed out during {operation}")
        return False

    def _lifecycle_failure(self, message: str) -> None:
        with self._condition:
            self._dropped["lifecycle"] += 1
            first = self._dropped["lifecycle"] == 1
        if self._strict:
            raise RecorderError(message)
        if first:
            _LOGGER.warning("trace recorder lifecycle operation rejected; see diagnostics")
        return None

    def _destination_failure(self, operation: str) -> None:
        with self._condition:
            self._dropped["destination_failures"] += 1
            first = self._dropped["destination_failures"] == 1
        if first:
            _LOGGER.warning("trace recorder failed to %s; see diagnostics", operation)

    def _bounded_metric_values(
        self, values: Mapping[str, int | float]
    ) -> Mapping[str, int | float]:
        selected = list(islice(values.items(), 65))
        if len(selected) > 64:
            self._dropped["attributes"] += 1
        result = {}
        for key, value in selected[:64]:
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("metric keys must be strings of at most 128 characters")
            if isinstance(value, int) and value.bit_length() > 256:
                raise ValueError("metric integer exceeds safety budget")
            result[key] = value
        return result


def _validate_correlation_id(value: str, field_name: str = "correlation_id") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > 256:
        raise ValueError(f"{field_name} must be at most 256 characters")


def _context_from_span(trace: Trace, span: Span, persisted: bool = True) -> _SpanContext:
    return _SpanContext(trace, span.span_id, span.execution_lane, span.status, persisted)


def _validate_nonempty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_timeout(timeout: float | None, field_name: str) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"{field_name} must be a number or None")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_cache_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("completed_context_cache_size must be an integer")
    if value <= 0:
        raise ValueError("completed_context_cache_size must be greater than zero")


def _deadline(timeout: float | None) -> float | None:
    _validate_timeout(timeout, "timeout")
    if timeout is None:
        return None
    return time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
