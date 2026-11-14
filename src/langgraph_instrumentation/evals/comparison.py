"""Baseline comparison with explicit quality/performance trade-offs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from ..models import JSONValue, attributes_to_dict, freeze_attributes
from .aggregation import (
    CaseAggregate,
    MetricAggregation,
    TrialAggregation,
    aggregate_run,
)
from .models import (
    EvalExecutionStatus,
    EvalMetric,
    EvalMetrics,
    EvalRun,
    MetricCompleteness,
    MissingMetricPolicy,
)

_COMPARISON_KIND = "langgraph_instrumentation.eval_comparison"
_SCHEMA_VERSION = "1"
_EMPTY_METADATA = MappingProxyType({})

DEFAULT_IDENTITY_KEYS = ("dataset.id", "agent.id", "eval.config", "model.id")


class MetricDirection(StrEnum):
    """Which direction of change is an improvement."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class ComparisonKind(StrEnum):
    """The deterministic classification of a case or metric change."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    ADDED = "added"
    REMOVED = "removed"
    INCOMPATIBLE = "incompatible"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MetricRule:
    """A named metric gate used when comparing runs."""

    metric: str
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    missing: MissingMetricPolicy = MissingMetricPolicy.FAIL

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric name must not be empty")
        try:
            direction = MetricDirection(self.direction)
        except ValueError as exc:
            raise ValueError("direction must be 'lower_is_better' or 'higher_is_better'") from exc
        try:
            missing = MissingMetricPolicy(self.missing)
        except ValueError as exc:
            raise ValueError("missing must be 'fail', 'error' or 'skip'") from exc
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "missing", missing)


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One metric's baseline/candidate values and classification."""

    metric: str
    kind: ComparisonKind
    direction: MetricDirection
    baseline: EvalMetric | None = None
    candidate: EvalMetric | None = None
    delta: float | None = None
    relative_delta: float | None = None
    tolerance: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if not isinstance(self.kind, ComparisonKind):
            raise TypeError("kind must be a ComparisonKind")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("direction must be a MetricDirection")
        for name, value in (
            ("baseline", self.baseline),
            ("candidate", self.candidate),
        ):
            if value is not None and not isinstance(value, EvalMetric):
                raise TypeError(f"{name} must be EvalMetric or null")

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "metric": self.metric,
                "kind": self.kind.value,
                "direction": self.direction.value,
                "baseline": self.baseline.to_dict() if self.baseline else None,
                "candidate": self.candidate.to_dict() if self.candidate else None,
                "delta": self.delta,
                "relative_delta": self.relative_delta,
                "tolerance": self.tolerance,
                "reason": self.reason,
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MetricComparison:
        return cls(
            metric=_required_string(data, "metric"),
            kind=ComparisonKind(_required_string(data, "kind")),
            direction=MetricDirection(_required_string(data, "direction")),
            baseline=_optional_metric(data.get("baseline")),
            candidate=_optional_metric(data.get("candidate")),
            delta=_optional_number(data, "delta"),
            relative_delta=_optional_number(data, "relative_delta"),
            tolerance=_optional_number(data, "tolerance"),
            reason=_optional_string(data, "reason"),
        )


@dataclass(frozen=True, slots=True)
class IdentityMismatch:
    """A dataset, agent or configuration value that differs between runs."""

    key: str
    baseline: JSONValue
    candidate: JSONValue

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("identity key must not be empty")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "key": self.key,
            "baseline": self.baseline,
            "candidate": self.candidate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IdentityMismatch:
        return cls(
            key=_required_string(data, "key"),
            baseline=_json_value(data.get("baseline")),
            candidate=_json_value(data.get("candidate")),
        )


@dataclass(frozen=True, slots=True)
class CaseComparison:
    """Quality and performance comparison for one case."""

    case_id: str
    kind: ComparisonKind
    quality: ComparisonKind
    metrics: tuple[MetricComparison, ...] = ()
    reasons: tuple[str, ...] = ()
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    baseline_status: EvalExecutionStatus | None = None
    candidate_status: EvalExecutionStatus | None = None
    baseline_trials: int = 0
    candidate_trials: int = 0
    baseline_execution_failures: int = 0
    candidate_execution_failures: int = 0
    baseline_assessment_errors: int = 0
    candidate_assessment_errors: int = 0

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not isinstance(self.kind, ComparisonKind):
            raise TypeError("kind must be a ComparisonKind")
        if not isinstance(self.quality, ComparisonKind):
            raise TypeError("quality must be a ComparisonKind")
        metrics = tuple(self.metrics)
        for metric in metrics:
            if not isinstance(metric, MetricComparison):
                raise TypeError("metrics must be MetricComparison records")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def passed(self) -> bool:
        return self.kind in (ComparisonKind.IMPROVED, ComparisonKind.UNCHANGED)

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "case_id": self.case_id,
                "kind": self.kind.value,
                "quality": self.quality.value,
                "metrics": [metric.to_dict() for metric in self.metrics],
                "reasons": list(self.reasons),
                "baseline_passed": self.baseline_passed,
                "candidate_passed": self.candidate_passed,
                "baseline_status": (
                    None if self.baseline_status is None else self.baseline_status.value
                ),
                "candidate_status": (
                    None if self.candidate_status is None else self.candidate_status.value
                ),
                "baseline_trials": self.baseline_trials,
                "candidate_trials": self.candidate_trials,
                "baseline_execution_failures": self.baseline_execution_failures,
                "candidate_execution_failures": self.candidate_execution_failures,
                "baseline_assessment_errors": self.baseline_assessment_errors,
                "candidate_assessment_errors": self.candidate_assessment_errors,
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CaseComparison:
        baseline_status = _optional_string(data, "baseline_status")
        candidate_status = _optional_string(data, "candidate_status")
        return cls(
            case_id=_required_string(data, "case_id"),
            kind=ComparisonKind(_required_string(data, "kind")),
            quality=ComparisonKind(_required_string(data, "quality")),
            metrics=tuple(
                MetricComparison.from_dict(_expect_mapping(item, "metric"))
                for item in _required_list(data, "metrics", default=[])
            ),
            reasons=tuple(str(item) for item in _required_list(data, "reasons", default=[])),
            baseline_passed=_optional_bool(data, "baseline_passed"),
            candidate_passed=_optional_bool(data, "candidate_passed"),
            baseline_status=(
                None if baseline_status is None else EvalExecutionStatus(baseline_status)
            ),
            candidate_status=(
                None if candidate_status is None else EvalExecutionStatus(candidate_status)
            ),
            baseline_trials=_optional_integer(data, "baseline_trials", 0),
            candidate_trials=_optional_integer(data, "candidate_trials", 0),
            baseline_execution_failures=_optional_integer(data, "baseline_execution_failures", 0),
            candidate_execution_failures=_optional_integer(data, "candidate_execution_failures", 0),
            baseline_assessment_errors=_optional_integer(data, "baseline_assessment_errors", 0),
            candidate_assessment_errors=_optional_integer(data, "candidate_assessment_errors", 0),
        )


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    """A versioned, deterministic baseline/candidate comparison artifact."""

    baseline_run_id: str
    candidate_run_id: str
    cases: tuple[CaseComparison, ...]
    trial_aggregation: TrialAggregation = TrialAggregation.WORST
    metric_aggregation: MetricAggregation = MetricAggregation.WORST
    identity_mismatches: tuple[IdentityMismatch, ...] = ()
    incompatible_reasons: tuple[str, ...] = ()
    require_same_cases: bool = True
    require_candidate_pass: bool = True
    schema_version: str = _SCHEMA_VERSION
    metadata: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_run_id", self.baseline_run_id),
            ("candidate_run_id", self.candidate_run_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison schema version: {self.schema_version!r}")
        cases = tuple(self.cases)
        identifiers: set[str] = set()
        for case in cases:
            if not isinstance(case, CaseComparison):
                raise TypeError("cases must be CaseComparison records")
            if case.case_id in identifiers:
                raise ValueError(f"duplicate case_id in comparison: {case.case_id!r}")
            identifiers.add(case.case_id)
        mismatches = tuple(self.identity_mismatches)
        for mismatch in mismatches:
            if not isinstance(mismatch, IdentityMismatch):
                raise TypeError("identity_mismatches must be IdentityMismatch records")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "identity_mismatches", mismatches)
        object.__setattr__(self, "incompatible_reasons", tuple(self.incompatible_reasons))
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))

    @property
    def passed(self) -> bool:
        if self.identity_mismatches or self.incompatible_reasons:
            return False
        if any(
            case.kind
            in (
                ComparisonKind.REGRESSED,
                ComparisonKind.INCOMPATIBLE,
                ComparisonKind.ERROR,
            )
            for case in self.cases
        ):
            return False
        if self.require_same_cases and any(
            case.kind in (ComparisonKind.ADDED, ComparisonKind.REMOVED) for case in self.cases
        ):
            return False
        return not (
            self.require_candidate_pass
            and any(case.candidate_passed is False for case in self.cases)
        )

    @property
    def regressed_case_ids(self) -> tuple[str, ...]:
        return tuple(
            case.case_id
            for case in self.cases
            if case.kind
            in (
                ComparisonKind.REGRESSED,
                ComparisonKind.INCOMPATIBLE,
                ComparisonKind.ERROR,
            )
        )

    @property
    def improved_case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases if case.kind is ComparisonKind.IMPROVED)

    @property
    def unchanged_case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases if case.kind is ComparisonKind.UNCHANGED)

    def case(self, case_id: str) -> CaseComparison | None:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "schema_version": self.schema_version,
                "kind": _COMPARISON_KIND,
                "baseline_run_id": self.baseline_run_id,
                "candidate_run_id": self.candidate_run_id,
                "trial_aggregation": self.trial_aggregation.value,
                "metric_aggregation": self.metric_aggregation.value,
                "require_same_cases": self.require_same_cases,
                "require_candidate_pass": self.require_candidate_pass,
                "passed": self.passed,
                "regressed_case_ids": list(self.regressed_case_ids),
                "improved_case_ids": list(self.improved_case_ids),
                "unchanged_case_ids": list(self.unchanged_case_ids),
                "identity_mismatches": [
                    mismatch.to_dict() for mismatch in self.identity_mismatches
                ],
                "incompatible_reasons": list(self.incompatible_reasons),
                "metadata": attributes_to_dict(self.metadata),
                "cases": [case.to_dict() for case in self.cases],
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ComparisonSummary:
        if data.get("kind") != _COMPARISON_KIND:
            raise ValueError("not an evaluation comparison artifact")
        version = _required_string(data, "schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison schema version: {version!r}")
        return cls(
            baseline_run_id=_required_string(data, "baseline_run_id"),
            candidate_run_id=_required_string(data, "candidate_run_id"),
            cases=tuple(
                CaseComparison.from_dict(_expect_mapping(item, "case"))
                for item in _required_list(data, "cases", default=[])
            ),
            trial_aggregation=TrialAggregation(_required_string(data, "trial_aggregation")),
            metric_aggregation=MetricAggregation(_required_string(data, "metric_aggregation")),
            identity_mismatches=tuple(
                IdentityMismatch.from_dict(_expect_mapping(item, "identity mismatch"))
                for item in _required_list(data, "identity_mismatches", default=[])
            ),
            incompatible_reasons=tuple(
                str(item) for item in _required_list(data, "incompatible_reasons", default=[])
            ),
            require_same_cases=_required_bool(data, "require_same_cases"),
            require_candidate_pass=_required_bool(data, "require_candidate_pass"),
            schema_version=version,
            metadata=_metadata(data.get("metadata")),
        )


def compare_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    rules: Sequence[MetricRule] = (),
    trials: TrialAggregation = TrialAggregation.WORST,
    metrics: MetricAggregation = MetricAggregation.WORST,
    require_same_cases: bool = True,
    require_candidate_pass: bool = True,
    identity_keys: Sequence[str] = DEFAULT_IDENTITY_KEYS,
    metadata: Mapping[str, object] | None = None,
) -> ComparisonSummary:
    """Compare two completed runs case by case.

    Quality regression is always a regression. Metric rules add explicit
    performance gates; no metric is compared implicitly. Dataset, agent and
    configuration identity keys must agree unless the caller changes the keys.
    """
    if not isinstance(baseline, EvalRun):
        raise TypeError("baseline must be an EvalRun")
    if not isinstance(candidate, EvalRun):
        raise TypeError("candidate must be an EvalRun")
    rule_list = tuple(rules)
    for rule in rule_list:
        if not isinstance(rule, MetricRule):
            raise TypeError("rules must contain MetricRule records")
    if not isinstance(trials, TrialAggregation):
        raise TypeError("trials must be a TrialAggregation")
    if not isinstance(metrics, MetricAggregation):
        raise TypeError("metrics must be a MetricAggregation")

    baseline_view = aggregate_run(baseline, trials=trials, metrics=metrics)
    candidate_view = aggregate_run(candidate, trials=trials, metrics=metrics)
    mismatches = tuple(
        mismatch
        for key in identity_keys
        if (mismatch := _identity_mismatch(key, baseline, candidate)) is not None
    )

    baseline_cases = {case.case_id: case for case in baseline_view.cases}
    candidate_cases = {case.case_id: case for case in candidate_view.cases}
    comparisons = tuple(
        _compare_case(case_id, baseline_cases.get(case_id), candidate_cases.get(case_id), rule_list)
        for case_id in sorted(set(baseline_cases) | set(candidate_cases))
    )
    return ComparisonSummary(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        cases=comparisons,
        trial_aggregation=trials,
        metric_aggregation=metrics,
        identity_mismatches=mismatches,
        require_same_cases=require_same_cases,
        require_candidate_pass=require_candidate_pass,
        metadata=metadata or {},
    )


def _compare_case(
    case_id: str,
    baseline: CaseAggregate | None,
    candidate: CaseAggregate | None,
    rules: tuple[MetricRule, ...],
) -> CaseComparison:
    if baseline is None:
        assert candidate is not None
        return CaseComparison(
            case_id=case_id,
            kind=ComparisonKind.ADDED,
            quality=ComparisonKind.ADDED,
            reasons=("case is not present in the baseline run",),
            candidate_passed=candidate.passed,
            candidate_trials=candidate.trial_count,
            candidate_execution_failures=candidate.execution_failures,
            candidate_assessment_errors=candidate.assessment_errors,
        )
    if candidate is None:
        return CaseComparison(
            case_id=case_id,
            kind=ComparisonKind.REMOVED,
            quality=ComparisonKind.REMOVED,
            reasons=("case is not present in the candidate run",),
            baseline_passed=baseline.passed,
            baseline_trials=baseline.trial_count,
            baseline_execution_failures=baseline.execution_failures,
            baseline_assessment_errors=baseline.assessment_errors,
        )

    quality = _quality_kind(baseline.passed, candidate.passed)
    metric_comparisons = tuple(
        _compare_metric(rule, baseline.metrics, candidate.metrics) for rule in rules
    )
    reasons: list[str] = []
    if quality is ComparisonKind.IMPROVED:
        reasons.append("case quality improved")
    elif quality is ComparisonKind.REGRESSED:
        reasons.append("case quality regressed")
    if candidate.execution_failures > baseline.execution_failures:
        reasons.append("candidate had more execution failures")
    if candidate.assessment_errors > baseline.assessment_errors:
        reasons.append("candidate had more assessment errors")
    reasons.extend(
        metric.reason
        for metric in metric_comparisons
        if metric.reason is not None and metric.kind is not ComparisonKind.UNCHANGED
    )

    kinds = [metric.kind for metric in metric_comparisons]
    if ComparisonKind.INCOMPATIBLE in kinds:
        kind = ComparisonKind.INCOMPATIBLE
    elif ComparisonKind.ERROR in kinds:
        kind = ComparisonKind.ERROR
    elif quality is ComparisonKind.REGRESSED or ComparisonKind.REGRESSED in kinds:
        kind = ComparisonKind.REGRESSED
    elif quality is ComparisonKind.IMPROVED or ComparisonKind.IMPROVED in kinds:
        kind = ComparisonKind.IMPROVED
    else:
        kind = ComparisonKind.UNCHANGED

    return CaseComparison(
        case_id=case_id,
        kind=kind,
        quality=quality,
        metrics=metric_comparisons,
        reasons=tuple(reasons),
        baseline_passed=baseline.passed,
        candidate_passed=candidate.passed,
        baseline_status=_dominant_status(baseline),
        candidate_status=_dominant_status(candidate),
        baseline_trials=baseline.trial_count,
        candidate_trials=candidate.trial_count,
        baseline_execution_failures=baseline.execution_failures,
        candidate_execution_failures=candidate.execution_failures,
        baseline_assessment_errors=baseline.assessment_errors,
        candidate_assessment_errors=candidate.assessment_errors,
    )


def _compare_metric(
    rule: MetricRule,
    baseline_metrics: EvalMetrics,
    candidate_metrics: EvalMetrics,
) -> MetricComparison:
    baseline = baseline_metrics.get(rule.metric)
    candidate = candidate_metrics.get(rule.metric)
    baseline_missing = baseline is None or baseline.completeness is MetricCompleteness.MISSING
    candidate_missing = candidate is None or candidate.completeness is MetricCompleteness.MISSING

    if baseline_missing or candidate_missing:
        if rule.missing is MissingMetricPolicy.SKIP:
            return MetricComparison(
                metric=rule.metric,
                kind=ComparisonKind.UNCHANGED,
                direction=rule.direction,
                baseline=baseline,
                candidate=candidate,
                reason=f"{rule.metric} is missing; comparison skipped by policy",
            )
        kind = (
            ComparisonKind.ERROR
            if rule.missing is MissingMetricPolicy.ERROR
            else ComparisonKind.INCOMPATIBLE
        )
        return MetricComparison(
            metric=rule.metric,
            kind=kind,
            direction=rule.direction,
            baseline=baseline,
            candidate=candidate,
            reason=f"{rule.metric} is missing in the baseline or candidate run",
        )

    assert baseline is not None and candidate is not None
    assert baseline.value is not None and candidate.value is not None
    delta = float(candidate.value) - float(baseline.value)
    tolerance = max(
        rule.absolute_tolerance,
        abs(float(baseline.value)) * rule.relative_tolerance,
    )
    if rule.direction is MetricDirection.LOWER_IS_BETTER:
        regressed = delta > tolerance
        improved = delta < -tolerance
    else:
        regressed = delta < -tolerance
        improved = delta > tolerance
    if regressed:
        kind = ComparisonKind.REGRESSED
        reason = (
            f"{rule.metric} changed by {delta:g} {baseline.unit}, "
            f"beyond the {tolerance:g} tolerance"
        )
    elif improved:
        kind = ComparisonKind.IMPROVED
        reason = (
            f"{rule.metric} changed by {delta:g} {baseline.unit}, "
            f"beyond the {tolerance:g} tolerance"
        )
    else:
        kind = ComparisonKind.UNCHANGED
        reason = None
    relative_delta = delta / abs(float(baseline.value)) if baseline.value != 0 else None
    return MetricComparison(
        metric=rule.metric,
        kind=kind,
        direction=rule.direction,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        relative_delta=relative_delta,
        tolerance=tolerance,
        reason=reason,
    )


def _quality_kind(baseline_passed: bool, candidate_passed: bool) -> ComparisonKind:
    if baseline_passed == candidate_passed:
        return ComparisonKind.UNCHANGED
    return ComparisonKind.IMPROVED if candidate_passed else ComparisonKind.REGRESSED


def _dominant_status(aggregate: CaseAggregate) -> EvalExecutionStatus | None:
    statuses = aggregate.execution_statuses
    if not statuses:
        return None
    for status in EvalExecutionStatus:
        if statuses.get(status.value) and status is not EvalExecutionStatus.COMPLETED:
            return status
    return EvalExecutionStatus.COMPLETED


def _identity_mismatch(key: str, baseline: EvalRun, candidate: EvalRun) -> IdentityMismatch | None:
    if key not in baseline.metadata or key not in candidate.metadata:
        return None
    baseline_value = _json_value(baseline.metadata[key])
    candidate_value = _json_value(candidate.metadata[key])
    if baseline_value == candidate_value:
        return None
    return IdentityMismatch(key=key, baseline=baseline_value, candidate=candidate_value)


def _json_value(value: object) -> JSONValue:
    return attributes_to_dict({"value": value})["value"]


def _optional_metric(value: object) -> EvalMetric | None:
    if value is None:
        return None
    return EvalMetric.from_dict(_expect_mapping(value, "metric"))


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _optional_bool(data: Mapping[str, object], key: str) -> bool | None:
    value = data.get(key)
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean or null")
    return value


def _optional_integer(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_number(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number or null")
    return float(value)


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _expect_mapping(data.get(key), key)


def _expect_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _required_list(
    data: Mapping[str, object], key: str, *, default: list[object] | None = None
) -> list[object]:
    value = data.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _metadata(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be an object")
    return freeze_attributes(value)
