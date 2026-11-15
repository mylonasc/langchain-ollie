"""Convert neutral trace evidence into attributable evaluation measurements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ..models import MetricPoint, Span, SpanKind, SpanStatus, TraceBundle
from .models import EvalMetric, EvalMetrics, MetricCompleteness, MetricScope

_TOKEN_METRIC = "llm.token.usage"
_PROCESS_METRIC = "process.resources"


def measure_traces(
    bundles: tuple[TraceBundle, ...],
    *,
    elapsed_ns: int,
    diagnostics: Mapping[str, int] | None = None,
) -> EvalMetrics:
    """Build turn measurements from recorded traces.

    Missing evidence is represented as ``None`` with ``MISSING`` completeness.
    Process measurements remain process-scoped; no per-agent attribution is made.
    """
    if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns < 0:
        raise ValueError("elapsed_ns must be a non-negative integer")

    metrics: dict[str, EvalMetric] = {
        "elapsed_ns": _metric("elapsed_ns", elapsed_ns, "ns", "runner.turn", MetricScope.TURN)
    }

    spans = tuple(span for bundle in bundles for span in bundle.spans)
    metrics["span.count"] = _count_metric("span.count", len(spans), MetricScope.TURN)
    error_spans = [span for span in spans if span.status is SpanStatus.ERROR]
    metrics["span.error_count"] = _count_metric(
        "span.error_count", len(error_spans), MetricScope.TURN
    )

    llm_spans = [span for span in spans if span.kind is SpanKind.LLM]
    tool_spans = [span for span in spans if span.kind is SpanKind.TOOL]
    metrics["llm.call_count"] = _count_metric("llm.call_count", len(llm_spans), MetricScope.TURN)
    metrics["tool.call_count"] = _count_metric("tool.call_count", len(tool_spans), MetricScope.TURN)
    metrics["tool.error_count"] = _count_metric(
        "tool.error_count",
        sum(span.status is SpanStatus.ERROR for span in tool_spans),
        MetricScope.TURN,
    )

    metrics["llm.duration_ns"] = _duration_metric("llm.duration_ns", llm_spans)
    metrics["tool.duration_ns"] = _duration_metric("tool.duration_ns", tool_spans)

    root_spans = [span for span in spans if span.parent_span_id is None]
    metrics["trace.duration_ns"] = _duration_metric(
        "trace.duration_ns",
        root_spans,
        detail="sum of root span durations",
    )

    metrics.update(_token_metrics(bundles, llm_spans))
    metrics.update(_process_metrics(bundles))
    metrics["recording.loss_count"] = _loss_metric(diagnostics or {})
    return EvalMetrics(metrics)


def aggregate_turn_metrics(turns: tuple[EvalMetrics, ...]) -> EvalMetrics:
    """Aggregate additive turn metrics into a case/trial metric set."""
    if not turns:
        return EvalMetrics()

    additive = (
        "elapsed_ns",
        "trace.duration_ns",
        "llm.call_count",
        "llm.duration_ns",
        "tool.call_count",
        "tool.error_count",
        "tool.duration_ns",
        "span.count",
        "span.error_count",
        "tokens.input",
        "tokens.output",
        "tokens.total",
        "recording.loss_count",
    )
    result: dict[str, EvalMetric] = {}
    for name in additive:
        available = [metrics.get(name) for metrics in turns]
        present = [metric for metric in available if metric is not None]
        if not present:
            continue
        missing = [
            metric for metric in present if metric.completeness is MetricCompleteness.MISSING
        ]
        partial = [
            metric for metric in present if metric.completeness is MetricCompleteness.PARTIAL
        ]
        if len(missing) == len(turns):
            result[name] = replace(
                present[0],
                value=None,
                completeness=MetricCompleteness.MISSING,
            )
            continue
        values = [
            metric.value
            for metric in present
            if metric.value is not None and metric.completeness is not MetricCompleteness.MISSING
        ]
        completeness = (
            MetricCompleteness.COMPLETE
            if len(present) == len(turns) and not missing and not partial
            else MetricCompleteness.PARTIAL
        )
        if len(missing) == len(turns):
            value: int | float | None = None
        else:
            value = sum(values)
        result[name] = replace(
            present[0],
            value=value,
            completeness=completeness,
            detail=(
                None
                if completeness is MetricCompleteness.COMPLETE
                else "aggregated across turns with missing or partial evidence"
            ),
            scope=MetricScope.CASE,
        )

    rss_values = [
        metric.value
        for metrics in turns
        for metric in (metrics.get("process.rss_bytes"),)
        if metric is not None and metric.value is not None
    ]
    if rss_values:
        last = next(
            metrics.get("process.rss_bytes")
            for metrics in reversed(turns)
            if metrics.get("process.rss_bytes") is not None
        )
        assert last is not None
        result["process.rss_bytes"] = replace(
            last,
            value=max(rss_values),
            scope=MetricScope.PROCESS,
            completeness=MetricCompleteness.PARTIAL,
            detail="process-wide sample maximum across turns; not per-agent memory",
        )

    for name in sorted(
        {
            metric_name
            for metrics in turns
            for metric_name in metrics
            if metric_name.startswith("process.") and metric_name != "process.rss_bytes"
        }
    ):
        present = [metrics.get(name) for metrics in turns]
        available = [metric for metric in present if metric is not None]
        if not available:
            continue
        usable = [
            metric
            for metric in available
            if metric.value is not None and metric.completeness is not MetricCompleteness.MISSING
        ]
        template = available[0]
        if not usable:
            result[name] = replace(
                template,
                value=None,
                scope=MetricScope.PROCESS,
                completeness=MetricCompleteness.MISSING,
                detail="all turns reported missing evidence",
            )
            continue
        total = sum(metric.value for metric in usable if metric.value is not None)
        complete = len(usable) == len(turns) and all(
            metric.completeness is MetricCompleteness.COMPLETE for metric in usable
        )
        result[name] = replace(
            template,
            value=total,
            scope=MetricScope.PROCESS,
            completeness=(MetricCompleteness.COMPLETE if complete else MetricCompleteness.PARTIAL),
            detail=(
                None
                if complete
                else (
                    f"{len(usable)} of {len(turns)} turn(s) contributed; "
                    "process-wide samples, not per-agent attribution"
                )
            ),
        )
    return EvalMetrics(result)


def _token_metrics(
    bundles: tuple[TraceBundle, ...], llm_spans: list[Span]
) -> dict[str, EvalMetric]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    traces_with_usage = 0
    for bundle in bundles:
        points = [metric for metric in bundle.metrics if metric.name == _TOKEN_METRIC]
        if not points:
            continue
        traces_with_usage += 1
        latest = max(
            points,
            key=lambda point: (point.time_monotonic_ns, str(point.span_id or "")),
        )
        for key in totals:
            totals[key] += _integer_value(latest, key)

    if not llm_spans:
        completeness = MetricCompleteness.COMPLETE
        detail = "no LLM calls in this turn"
    elif traces_with_usage == len(bundles) and llm_spans:
        completeness = MetricCompleteness.COMPLETE
        detail = "provider-reported usage from the final LLM call in each trace"
    elif traces_with_usage:
        completeness = MetricCompleteness.PARTIAL
        detail = "provider usage was not reported by every trace"
    else:
        completeness = MetricCompleteness.MISSING
        detail = "provider usage was not reported; no estimate is substituted"

    metrics: dict[str, EvalMetric] = {}
    mapping = {
        "tokens.input": ("input_tokens", "tokens"),
        "tokens.output": ("output_tokens", "tokens"),
        "tokens.total": ("total_tokens", "tokens"),
    }
    for name, (key, unit) in mapping.items():
        metrics[name] = EvalMetric(
            name=name,
            value=totals[key] if completeness is not MetricCompleteness.MISSING else None,
            unit=unit,
            source="trace.metrics",
            scope=MetricScope.TURN,
            completeness=completeness,
            detail=detail,
        )
    return metrics


def _process_metrics(bundles: tuple[TraceBundle, ...]) -> dict[str, EvalMetric]:
    points = [
        metric for bundle in bundles for metric in bundle.metrics if metric.name == _PROCESS_METRIC
    ]
    if not points:
        missing = EvalMetric(
            name="process.samples",
            value=None,
            unit="samples",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=MetricCompleteness.MISSING,
            detail="no process sampler was configured for this turn",
        )
        return {"process.samples": missing}

    ordered = sorted(
        points,
        key=lambda point: (point.time_monotonic_ns, str(point.span_id or "")),
    )
    first_cpu = ordered[0].values.get("cpu_seconds")
    last_cpu = ordered[-1].values.get("cpu_seconds")
    rss = [point.values.get("rss_bytes") for point in ordered]
    rss_values = [value for value in rss if isinstance(value, (int, float))]
    last = ordered[-1]
    metrics = {
        "process.samples": EvalMetric(
            name="process.samples",
            value=len(ordered),
            unit="samples",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=MetricCompleteness.PARTIAL,
            detail="sampled process-wide measurements; not per-agent attribution",
        ),
        "process.rss_bytes": EvalMetric(
            name="process.rss_bytes",
            value=max(rss_values) if rss_values else None,
            unit="bytes",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=(MetricCompleteness.PARTIAL if rss_values else MetricCompleteness.MISSING),
            detail="process-wide RSS sample maximum; not per-agent memory",
        ),
    }
    if isinstance(first_cpu, (int, float)) and isinstance(last_cpu, (int, float)):
        metrics["process.cpu_seconds"] = EvalMetric(
            name="process.cpu_seconds",
            value=max(0.0, float(last_cpu) - float(first_cpu)),
            unit="s",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=MetricCompleteness.PARTIAL,
            detail="process-wide CPU delta between samples; not wall time",
        )
    else:
        metrics["process.cpu_seconds"] = EvalMetric(
            name="process.cpu_seconds",
            value=None,
            unit="s",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=MetricCompleteness.MISSING,
            detail="CPU samples were unavailable",
        )
    for key, name in (
        ("missing_samples", "process.missing_samples"),
        ("dropped_samples", "process.dropped_samples"),
    ):
        value = last.values.get(key)
        metrics[name] = EvalMetric(
            name=name,
            value=int(value) if isinstance(value, (int, float)) else 0,
            unit="samples",
            source="process sampler",
            scope=MetricScope.PROCESS,
            completeness=(
                MetricCompleteness.COMPLETE
                if isinstance(value, (int, float))
                else MetricCompleteness.MISSING
            ),
            detail="reported by the optional process sampler",
        )
    return metrics


def _loss_metric(diagnostics: Mapping[str, int]) -> EvalMetric:
    lossy = {name: value for name, value in diagnostics.items() if value}
    return EvalMetric(
        name="recording.loss_count",
        value=sum(lossy.values()),
        unit="observations",
        source="recording diagnostics",
        scope=MetricScope.TURN,
        completeness=(MetricCompleteness.COMPLETE if not lossy else MetricCompleteness.PARTIAL),
        detail=(
            "no recording loss reported"
            if not lossy
            else "loss reported by: " + ", ".join(sorted(lossy))
        ),
    )


def _duration_metric(name: str, spans: list[Span], *, detail: str | None = None) -> EvalMetric:
    finished = [span for span in spans if span.duration_ns is not None]
    if len(finished) != len(spans):
        return EvalMetric(
            name=name,
            value=sum(span.duration_ns or 0 for span in finished),
            unit="ns",
            source="trace.spans",
            scope=MetricScope.TURN,
            completeness=MetricCompleteness.PARTIAL,
            detail="some spans were unfinished or abandoned",
        )
    return EvalMetric(
        name=name,
        value=sum(span.duration_ns or 0 for span in finished),
        unit="ns",
        source="trace.spans",
        scope=MetricScope.TURN,
        completeness=MetricCompleteness.COMPLETE,
        detail=detail,
    )


def _count_metric(name: str, value: int, scope: MetricScope) -> EvalMetric:
    return EvalMetric(
        name=name,
        value=value,
        unit="count",
        source="trace.spans",
        scope=scope,
    )


def _metric(
    name: str,
    value: int | float,
    unit: str,
    source: str,
    scope: MetricScope,
    *,
    detail: str | None = None,
) -> EvalMetric:
    return EvalMetric(
        name=name,
        value=value,
        unit=unit,
        source=source,
        scope=scope,
        detail=detail,
    )


def _integer_value(point: MetricPoint, key: str) -> int:
    value = point.values.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)
