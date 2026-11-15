"""Bounded, read-only inspection over evaluation runs and comparisons.

These operations consume plain run artifacts and caller-supplied trace bundles
only. They never import store implementations, query databases, or retain
application payloads beyond explicit truncation budgets. Missing evidence is
reported, never synthesized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..models import JSONValue, SpanKind, TraceBundle
from .comparison import ComparisonSummary
from .models import EvalMetrics, EvalRun, EvalTrial, EvalTurnResult

_INSPECT_KIND_FAILURES = "langgraph_instrumentation.failure_summary"
_INSPECT_KIND_TURN = "langgraph_instrumentation.turn_evidence"
_INSPECT_KIND_REGRESSION = "langgraph_instrumentation.regression_summary"
_SCHEMA_VERSION = "1"

_DEFAULT_MAX_CASES = 20
_DEFAULT_MAX_REASONS = 5
_DEFAULT_MAX_TEXT = 2_000
_DEFAULT_MAX_CALLS = 50


@dataclass(frozen=True, slots=True)
class FailureCase:
    """One failed trial with bounded, actionable reasons."""

    case_id: str
    trial_index: int
    execution_status: str
    passed: bool = False
    reasons: tuple[str, ...] = ()
    reasons_truncated: bool = False
    trace_ids: tuple[str, ...] = ()
    error_type: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "trial_index": self.trial_index,
            "execution_status": self.execution_status,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "reasons_truncated": self.reasons_truncated,
            "trace_ids": list(self.trace_ids),
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class FailureSummary:
    """Concise versioned summary of every failed trial in a run."""

    run_id: str
    total_trials: int
    failed_trials: int
    failed_cases: tuple[FailureCase, ...] = ()
    truncated: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported failure summary version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _INSPECT_KIND_FAILURES,
            "run_id": self.run_id,
            "total_trials": self.total_trials,
            "failed_trials": self.failed_trials,
            "truncated": self.truncated,
            "failed_cases": [case.to_dict() for case in self.failed_cases],
        }


@dataclass(frozen=True, slots=True)
class ObservedCall:
    """One tool or LLM call observed in caller-supplied trace evidence."""

    name: str
    kind: str
    status: str
    duration_ns: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "duration_ns": self.duration_ns,
        }


@dataclass(frozen=True, slots=True)
class TurnEvidence:
    """Bounded evidence for one turn: output, assessments, metrics and calls."""

    case_id: str
    turn_index: int
    execution_status: str
    output_text: str | None = None
    output_truncated: bool = False
    assessments: tuple[Mapping[str, JSONValue], ...] = ()
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    trace_ids: tuple[str, ...] = ()
    missing_trace_ids: tuple[str, ...] = ()
    calls: tuple[ObservedCall, ...] = ()
    calls_truncated: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported turn evidence version: {self.schema_version!r}")
        object.__setattr__(self, "assessments", tuple(self.assessments))
        object.__setattr__(self, "trace_ids", tuple(self.trace_ids))
        object.__setattr__(self, "missing_trace_ids", tuple(self.missing_trace_ids))
        object.__setattr__(self, "calls", tuple(self.calls))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _INSPECT_KIND_TURN,
            "case_id": self.case_id,
            "turn_index": self.turn_index,
            "execution_status": self.execution_status,
            "output_text": self.output_text,
            "output_truncated": self.output_truncated,
            "assessments": [dict(item) for item in self.assessments],
            "metrics": self.metrics.to_dict(),
            "trace_ids": list(self.trace_ids),
            "missing_trace_ids": list(self.missing_trace_ids),
            "calls": [call.to_dict() for call in self.calls],
            "calls_truncated": self.calls_truncated,
        }


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """One compared case with its quality outcome and metric deltas."""

    case_id: str
    kind: str
    quality: str
    reasons: tuple[str, ...] = ()
    metric_deltas: tuple[Mapping[str, JSONValue], ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "quality": self.quality,
            "reasons": list(self.reasons),
            "metric_deltas": [dict(item) for item in self.metric_deltas],
        }


@dataclass(frozen=True, slots=True)
class RegressionSummary:
    """Concise versioned summary of a baseline comparison."""

    baseline_run_id: str
    candidate_run_id: str
    passed: bool
    regressed_case_ids: tuple[str, ...] = ()
    improved_case_ids: tuple[str, ...] = ()
    unchanged_case_ids: tuple[str, ...] = ()
    cases: tuple[RegressionCase, ...] = ()
    truncated: bool = False
    identity_mismatches: tuple[Mapping[str, JSONValue], ...] = ()
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported regression summary version: {self.schema_version!r}")
        object.__setattr__(self, "regressed_case_ids", tuple(self.regressed_case_ids))
        object.__setattr__(self, "improved_case_ids", tuple(self.improved_case_ids))
        object.__setattr__(self, "unchanged_case_ids", tuple(self.unchanged_case_ids))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "identity_mismatches", tuple(self.identity_mismatches))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _INSPECT_KIND_REGRESSION,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "passed": self.passed,
            "regressed_case_ids": list(self.regressed_case_ids),
            "improved_case_ids": list(self.improved_case_ids),
            "unchanged_case_ids": list(self.unchanged_case_ids),
            "truncated": self.truncated,
            "identity_mismatches": [dict(item) for item in self.identity_mismatches],
            "cases": [case.to_dict() for case in self.cases],
        }


def summarize_failures(run: EvalRun, *, max_cases: int = _DEFAULT_MAX_CASES) -> FailureSummary:
    """Summarize every failed trial; bound the output and flag truncation."""
    if not isinstance(run, EvalRun):
        raise TypeError("run must be an EvalRun")
    _require_positive(max_cases, "max_cases")
    failed = [trial for trial in run.trials if not trial.passed]
    selected = failed[:max_cases]
    return FailureSummary(
        run_id=run.run_id,
        total_trials=len(run.trials),
        failed_trials=len(failed),
        failed_cases=tuple(_failure_case(trial) for trial in selected),
        truncated=len(failed) > len(selected),
    )


def turn_evidence(
    run: EvalRun,
    case_id: str,
    turn_index: int,
    *,
    traces: Mapping[str, TraceBundle] | None = None,
    max_text: int = _DEFAULT_MAX_TEXT,
    max_calls: int = _DEFAULT_MAX_CALLS,
) -> TurnEvidence:
    """Collect bounded evidence for one turn.

    Trace bundles are caller-supplied; trace IDs without a matching bundle are
    reported as missing, never synthesized.
    """
    if not isinstance(run, EvalRun):
        raise TypeError("run must be an EvalRun")
    _require_positive(max_text, "max_text")
    _require_positive(max_calls, "max_calls")
    bundles = dict(traces or {})
    for key, bundle in bundles.items():
        if not isinstance(bundle, TraceBundle):
            raise TypeError(f"trace bundle {key!r} must be a TraceBundle")
    turn = _find_turn(run, case_id, turn_index)
    text, truncated = _truncate(
        turn.output.text if turn.output and turn.output.text else None, max_text
    )
    present = [bundles[trace_id] for trace_id in turn.trace_ids if trace_id in bundles]
    missing = tuple(trace_id for trace_id in turn.trace_ids if trace_id not in bundles)
    calls = _observed_calls(present)
    assessments: list[Mapping[str, JSONValue]] = [
        {
            "evaluator": assessment.evaluator,
            "status": assessment.status.value,
            "message": assessment.message,
        }
        for assessment in turn.assessments
    ]
    return TurnEvidence(
        case_id=case_id,
        turn_index=turn_index,
        execution_status=turn.execution_status.value,
        output_text=text,
        output_truncated=truncated,
        assessments=tuple(assessments),
        metrics=turn.metrics,
        trace_ids=turn.trace_ids,
        missing_trace_ids=missing,
        calls=tuple(calls[:max_calls]),
        calls_truncated=len(calls) > max_calls,
    )


def summarize_regression(
    comparison: ComparisonSummary, *, max_cases: int = _DEFAULT_MAX_CASES
) -> RegressionSummary:
    """Summarize a comparison for agents: outcomes, deltas and mismatches."""
    if not isinstance(comparison, ComparisonSummary):
        raise TypeError("comparison must be a ComparisonSummary")
    _require_positive(max_cases, "max_cases")
    selected = comparison.cases[:max_cases]
    summarized: list[RegressionCase] = []
    for case in selected:
        deltas: list[Mapping[str, JSONValue]] = [
            {
                "metric": metric.metric,
                "kind": metric.kind.value,
                "delta": metric.delta,
                "reason": metric.reason,
            }
            for metric in case.metrics
        ]
        summarized.append(
            RegressionCase(
                case_id=case.case_id,
                kind=case.kind.value,
                quality=case.quality.value,
                reasons=tuple(case.reasons),
                metric_deltas=tuple(deltas),
            )
        )
    mismatches: list[Mapping[str, JSONValue]] = [
        mismatch.to_dict() for mismatch in comparison.identity_mismatches
    ]
    return RegressionSummary(
        baseline_run_id=comparison.baseline_run_id,
        candidate_run_id=comparison.candidate_run_id,
        passed=comparison.passed,
        regressed_case_ids=comparison.regressed_case_ids,
        improved_case_ids=comparison.improved_case_ids,
        unchanged_case_ids=comparison.unchanged_case_ids,
        cases=tuple(summarized),
        truncated=len(comparison.cases) > len(selected),
        identity_mismatches=tuple(mismatches),
    )


def _failure_case(trial: EvalTrial) -> FailureCase:
    reasons: list[str] = []
    assessments = [
        *trial.assessments,
        *(item for turn in trial.turns for item in turn.assessments),
    ]
    for assessment in assessments:
        if assessment.status.value in ("fail", "error"):
            reasons.append(
                f"{assessment.evaluator}: {assessment.message or assessment.status.value}"
            )
    if trial.error is not None:
        reasons.append(f"execution: {trial.error.exception_type}")
    for turn in trial.turns:
        if turn.execution_status.value != "completed":
            reasons.append(f"turn {turn.index}: {turn.execution_status.value}")
    selected = reasons[:_DEFAULT_MAX_REASONS]
    return FailureCase(
        case_id=trial.case_id,
        trial_index=trial.trial_index,
        execution_status=trial.execution_status.value,
        passed=trial.passed,
        reasons=tuple(selected),
        reasons_truncated=len(reasons) > len(selected),
        trace_ids=trial.trace_ids,
        error_type=trial.error.exception_type if trial.error else None,
    )


def _find_turn(run: EvalRun, case_id: str, turn_index: int) -> EvalTurnResult:
    for trial in run.trials:
        if trial.case_id != case_id:
            continue
        for turn in trial.turns:
            if turn.index == turn_index:
                return turn
    raise KeyError(f"no turn {turn_index} for case {case_id!r} in run {run.run_id!r}")


def _truncate(text: str | None, limit: int) -> tuple[str | None, bool]:
    if text is None or len(text) <= limit:
        return text, False
    return text[:limit] + f"<truncated:{len(text) - limit}>", True


def _observed_calls(bundles: list[TraceBundle]) -> list[ObservedCall]:
    calls: list[ObservedCall] = []
    for bundle in bundles:
        for span in bundle.spans:
            if span.kind is SpanKind.LLM:
                calls.append(
                    ObservedCall(
                        name=span.name,
                        kind="llm",
                        status=span.status.value,
                        duration_ns=span.duration_ns,
                    )
                )
            elif span.kind is SpanKind.TOOL:
                calls.append(
                    ObservedCall(
                        name=span.name,
                        kind="tool",
                        status=span.status.value,
                        duration_ns=span.duration_ns,
                    )
                )
    calls.sort(key=lambda call: (call.kind, call.name, call.duration_ns or 0))
    return calls


def _require_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
