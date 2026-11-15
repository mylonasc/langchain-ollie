"""Schema-independent shared store validation; no concrete implementation imports."""

import math
from dataclasses import replace

from ..models import Span, SpanStatus, Trace


def _active_trace(trace: Trace) -> Trace:
    return trace if trace.end_time_unix_ns is None else replace(trace, end_time_unix_ns=None)


def _same_started_span(active: Span, completed: Span) -> bool:
    return (
        replace(
            completed,
            end_time_unix_ns=None,
            end_time_monotonic_ns=None,
            status=SpanStatus.UNSET,
            status_description=None,
            attributes=active.attributes,
            events=active.events,
        )
        == active
    )


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number or None")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and non-negative")
