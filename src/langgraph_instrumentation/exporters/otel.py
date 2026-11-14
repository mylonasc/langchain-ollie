"""Optional export-only OpenTelemetry SDK consumer. No global provider changes."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags, TraceState
from opentelemetry.trace import SpanKind as OTelSpanKind

from ..models import Span, SpanStatus, attributes_to_dict


def _attributes(values: Mapping[str, object]) -> dict[str, str | bool | int | float]:
    """Preserve neutral nested data as JSON rather than invalid SDK attribute types."""
    result: dict[str, str | bool | int | float] = {}
    for key, value in attributes_to_dict(values).items():
        if isinstance(value, (str, bool, int, float)):
            result[key] = value
        else:
            result[key] = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result


def _context(trace_id: int, span_id: int) -> SpanContext:
    # Export-only records have no incoming trace flags/tracestate or remote parent.
    return SpanContext(trace_id, span_id, False, TraceFlags(TraceFlags.SAMPLED), TraceState())


def to_otel_span(span: Span) -> ReadableSpan:
    """Convert completed neutral records without generating IDs or replaying timing."""
    if not span.is_finished:
        raise ValueError("OpenTelemetry conversion requires a completed span")
    attributes = _attributes(span.attributes)
    attributes.update(
        {
            "local_trace.span.kind": span.kind.value,
            "local_trace.span.status": span.status.value,
            "local_trace.schema.version": 1,
            "local_trace.context.mode": "export_only",
        }
    )
    code = {SpanStatus.SUCCESS: StatusCode.OK, SpanStatus.ERROR: StatusCode.ERROR}.get(
        span.status, StatusCode.UNSET
    )
    # Error descriptions may contain application payload; keep them local only.
    links = [
        Link(
            _context(link.trace_id.value, link.span_id.value),
            {**_attributes(link.attributes), "local_trace.relationship": link.relationship},
        )
        for link in span.links
    ]
    return ReadableSpan(
        name=span.name,
        context=_context(span.trace_id.value, span.span_id.value),
        parent=_context(span.trace_id.value, span.parent_span_id.value)
        if span.parent_span_id
        else None,
        resource=Resource(_attributes(span.resource_attributes)),
        attributes=attributes,
        events=[Event(e.name, _attributes(e.attributes), e.time_unix_ns) for e in span.events],
        links=links,
        kind=OTelSpanKind.INTERNAL,
        status=Status(code),
        start_time=span.start_time_unix_ns,
        end_time=span.end_time_unix_ns,
        instrumentation_scope=InstrumentationScope("langgraph_instrumentation", "0.1.0"),
    )


class _ObservedExporter(SpanExporter):
    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter
        self.failures = 0
        self._lock = threading.Lock()

    def _failed(self) -> None:
        with self._lock:
            self.failures += 1

    def export(self, spans) -> SpanExportResult:
        try:
            result = self.exporter.export(spans)
        except Exception:
            self._failed()  # do not let SDK exception logging retain destination bodies
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            self._failed()
        return result

    def shutdown(self) -> None:
        try:
            self.exporter.shutdown()
        except Exception:
            self._failed()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return bool(self.exporter.force_flush(timeout_millis))
        except Exception:
            self._failed()
            return False


class OpenTelemetryProcessor:
    """Export-only neutral consumer using standard SDK batching and exporters.

    Owns the supplied exporter. Default batching is bounded by the SDK queue;
    failures are observable but do not erase local traces. No provider, sampler,
    propagator, environment exporter, or network endpoint is configured implicitly.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        batch: bool = True,
        max_queue_size: int = 2048,
        max_export_batch_size: int = 128,
    ) -> None:
        if (
            type(max_queue_size) is not int
            or type(max_export_batch_size) is not int
            or not 1 <= max_export_batch_size <= max_queue_size <= 100_000
        ):
            raise ValueError("require 1 <= batch size <= queue size <= 100000")
        self._exporter = _ObservedExporter(exporter)
        self._processor = (
            BatchSpanProcessor(
                self._exporter,
                max_queue_size=max_queue_size,
                max_export_batch_size=max_export_batch_size,
                schedule_delay_millis=5000,
                export_timeout_millis=30000,
            )
            if batch
            else SimpleSpanProcessor(self._exporter)
        )
        self._shutdown_thread: threading.Thread | None = None
        self._flush_thread: threading.Thread | None = None
        self._flush_result = False
        self._lock = threading.Lock()

    @property
    def diagnostics(self) -> dict[str, int]:
        return {"export_failures": self._exporter.failures}

    def on_end(self, span: Span) -> None:
        if self._shutdown_thread is not None:
            raise RuntimeError("OpenTelemetry processor is shutting down")
        self._processor.on_end(to_otel_span(span))

    @staticmethod
    def _milliseconds(timeout: float | None) -> int:
        if timeout is None:
            return 30000
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        return int(timeout * 1000)

    def force_flush(self, timeout: float | None = None) -> bool:
        milliseconds = self._milliseconds(timeout)
        with self._lock:
            if self._shutdown_thread is not None:
                return False
            if self._flush_thread is None or not self._flush_thread.is_alive():
                self._flush_result = False

                def flush() -> None:
                    try:
                        self._flush_result = self._processor.force_flush(milliseconds)
                    except Exception:
                        self._exporter._failed()

                self._flush_thread = threading.Thread(
                    target=flush, name="local-trace-otel-flush", daemon=True
                )
                self._flush_thread.start()
            worker = self._flush_thread
        # Some SDK versions ignore the supplied force_flush timeout. One shared
        # worker bounds caller wait without adding a transport/retry queue.
        worker.join(timeout)
        return not worker.is_alive() and self._flush_result and self._exporter.failures == 0

    def shutdown(self, timeout: float | None = None) -> bool:
        self._milliseconds(timeout)
        start = time.monotonic()
        with self._lock:
            if self._shutdown_thread is None:
                self._shutdown_thread = threading.Thread(
                    target=self._processor.shutdown, name="local-trace-otel-shutdown", daemon=True
                )
                self._shutdown_thread.start()
            worker = self._shutdown_thread
        # SDK shutdown has no public timeout argument. Bound the caller's wait, not
        # a foreign callback. A later call waits for the same cooperative cleanup.
        worker.join(timeout)
        flush = self._flush_thread
        if flush is not None:
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
            flush.join(remaining)
        return (
            not worker.is_alive()
            and (flush is None or not flush.is_alive())
            and self._exporter.failures == 0
        )
