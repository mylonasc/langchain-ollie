"""Fail-safe aggregation of repeated evaluation trials."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from ..models import JSONValue, attributes_to_dict, freeze_attributes
from .models import (
    EvalExecutionStatus,
    EvalMetric,
    EvalMetrics,
    EvalRun,
    EvalStatus,
    EvalTrial,
    MetricCompleteness,
    MetricScope,
)

_AGGREGATE_KIND = "langgraph_instrumentation.eval_aggregate"
_SCHEMA_VERSION = "1"
_EMPTY_METRICS = MappingProxyType({})


class TrialAggregation(StrEnum):
    """How repeated trials determine a case outcome."""

    WORST = "worst"
    BEST = "best"
    MAJORITY = "majority"


class MetricAggregation(StrEnum):
    """How repeated trial measurements combine.

    ``WORST`` selects the maximum, which is conservative for cost and latency
    metrics. ``BEST`` selects the minimum. Callers choose explicitly because no
    single direction is meaningful for every metric.
    """

    WORST = "worst"
    BEST = "best"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class CaseAggregate:
    """The fail-safe outcome and measurements of one case across its trials."""

    case_id: str
    trial_count: int
    passed_trials: int
    execution_failures: int
    assessment_errors: int
    passed: bool
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    trial_indices: tuple[int, ...] = ()
    execution_statuses: Mapping[str, int] = field(default_factory=lambda: _EMPTY_METRICS)
    missing_metric_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        for name, value in (
            ("trial_count", self.trial_count),
            ("passed_trials", self.passed_trials),
            ("execution_failures", self.execution_failures),
            ("assessment_errors", self.assessment_errors),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.passed_trials > self.trial_count:
            raise ValueError("passed_trials cannot exceed trial_count")
        if self.execution_failures > self.trial_count:
            raise ValueError("execution_failures cannot exceed trial_count")
        if not isinstance(self.metrics, EvalMetrics):
            raise TypeError("metrics must be EvalMetrics")
        indices = tuple(self.trial_indices)
        if any(isinstance(index, bool) or index < 0 for index in indices):
            raise ValueError("trial indices must be non-negative integers")
        statuses = {
            str(name): int(value) for name, value in sorted(self.execution_statuses.items())
        }
        missing = tuple(sorted(set(self.missing_metric_names)))
        object.__setattr__(self, "trial_indices", indices)
        object.__setattr__(self, "execution_statuses", MappingProxyType(statuses))
        object.__setattr__(self, "missing_metric_names", missing)

    @property
    def pass_rate(self) -> float:
        return self.passed_trials / self.trial_count if self.trial_count else 0.0

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "case_id": self.case_id,
                "trial_count": self.trial_count,
                "passed_trials": self.passed_trials,
                "execution_failures": self.execution_failures,
                "assessment_errors": self.assessment_errors,
                "passed": self.passed,
                "pass_rate": self.pass_rate,
                "metrics": self.metrics.to_dict(),
                "trial_indices": list(self.trial_indices),
                "execution_statuses": dict(self.execution_statuses),
                "missing_metric_names": list(self.missing_metric_names),
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CaseAggregate:
        statuses = data.get("execution_statuses", {})
        if not isinstance(statuses, Mapping):
            raise TypeError("execution_statuses must be an object")
        return cls(
            case_id=_required_string(data, "case_id"),
            trial_count=_required_integer(data, "trial_count"),
            passed_trials=_required_integer(data, "passed_trials"),
            execution_failures=_required_integer(data, "execution_failures"),
            assessment_errors=_required_integer(data, "assessment_errors"),
            passed=_required_bool(data, "passed"),
            metrics=EvalMetrics.from_dict(_expect_mapping(data.get("metrics"), "metrics")),
            trial_indices=tuple(
                _as_int(item, "trial_indices")
                for item in _required_list(data, "trial_indices", default=[])
            ),
            execution_statuses={
                str(key): _as_int(value, "execution_statuses") for key, value in statuses.items()
            },
            missing_metric_names=tuple(
                str(item) for item in _required_list(data, "missing_metric_names", default=[])
            ),
        )


@dataclass(frozen=True, slots=True)
class RunAggregate:
    """A deterministic case-level view over repeated trials in one run."""

    run_id: str
    cases: tuple[CaseAggregate, ...]
    trial_aggregation: TrialAggregation = TrialAggregation.WORST
    metric_aggregation: MetricAggregation = MetricAggregation.WORST
    schema_version: str = _SCHEMA_VERSION
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(self.trial_aggregation, TrialAggregation):
            raise TypeError("trial_aggregation must be a TrialAggregation")
        if not isinstance(self.metric_aggregation, MetricAggregation):
            raise TypeError("metric_aggregation must be a MetricAggregation")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported aggregate schema version: {self.schema_version!r}")
        cases = tuple(self.cases)
        identifiers: set[str] = set()
        for case in cases:
            if not isinstance(case, CaseAggregate):
                raise TypeError("cases must be CaseAggregate records")
            if case.case_id in identifiers:
                raise ValueError(f"duplicate case_id in aggregate: {case.case_id!r}")
            identifiers.add(case.case_id)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def case(self, case_id: str) -> CaseAggregate | None:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "schema_version": self.schema_version,
                "kind": _AGGREGATE_KIND,
                "run_id": self.run_id,
                "trial_aggregation": self.trial_aggregation.value,
                "metric_aggregation": self.metric_aggregation.value,
                "metadata": attributes_to_dict(self.metadata),
                "cases": [case.to_dict() for case in self.cases],
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunAggregate:
        if data.get("kind") != _AGGREGATE_KIND:
            raise ValueError("not an evaluation aggregate artifact")
        version = _required_string(data, "schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported aggregate schema version: {version!r}")
        return cls(
            run_id=_required_string(data, "run_id"),
            cases=tuple(
                CaseAggregate.from_dict(_expect_mapping(item, "case"))
                for item in _required_list(data, "cases", default=[])
            ),
            trial_aggregation=TrialAggregation(_required_string(data, "trial_aggregation")),
            metric_aggregation=MetricAggregation(_required_string(data, "metric_aggregation")),
            metadata=_metadata(data.get("metadata")),
            schema_version=version,
        )


def aggregate_run(
    run: EvalRun,
    *,
    trials: TrialAggregation = TrialAggregation.WORST,
    metrics: MetricAggregation = MetricAggregation.WORST,
    metadata: Mapping[str, object] | None = None,
) -> RunAggregate:
    """Aggregate all trials of every case, sorted by case ID.

    The default trial policy is fail-safe: a case passes only when every trial
    passes. The default metric policy selects the maximum value.
    """
    if not isinstance(run, EvalRun):
        raise TypeError("run must be an EvalRun")
    if not isinstance(trials, TrialAggregation):
        raise TypeError("trials must be a TrialAggregation")
    if not isinstance(metrics, MetricAggregation):
        raise TypeError("metrics must be a MetricAggregation")

    grouped: dict[str, list[EvalTrial]] = {}
    for trial in run.trials:
        grouped.setdefault(trial.case_id, []).append(trial)

    cases = tuple(
        _aggregate_case(case_id, ordered, trials, metrics)
        for case_id, ordered in sorted(grouped.items())
    )
    return RunAggregate(
        run_id=run.run_id,
        cases=cases,
        trial_aggregation=trials,
        metric_aggregation=metrics,
        metadata=metadata or {},
    )


def _aggregate_case(
    case_id: str,
    trials: list[EvalTrial],
    trial_policy: TrialAggregation,
    metric_policy: MetricAggregation,
) -> CaseAggregate:
    ordered = sorted(trials, key=lambda trial: trial.trial_index)
    passed_trials = sum(trial.passed for trial in ordered)
    if trial_policy is TrialAggregation.WORST:
        passed = passed_trials == len(ordered)
    elif trial_policy is TrialAggregation.BEST:
        passed = passed_trials > 0
    else:
        passed = 2 * passed_trials > len(ordered)

    execution_failures = sum(
        trial.execution_status is not EvalExecutionStatus.COMPLETED for trial in ordered
    )
    assessment_errors = 0
    for trial in ordered:
        assessment_errors += sum(
            assessment.status is EvalStatus.ERROR for assessment in trial.assessments
        )
        for turn in trial.turns:
            assessment_errors += sum(
                assessment.status is EvalStatus.ERROR for assessment in turn.assessments
            )
    statuses: dict[str, int] = {}
    for trial in ordered:
        key = trial.execution_status.value
        statuses[key] = statuses.get(key, 0) + 1

    metrics = _aggregate_metrics(ordered, metric_policy)
    missing = tuple(
        name
        for name, metric in metrics.values.items()
        if metric.completeness is MetricCompleteness.MISSING
    )
    return CaseAggregate(
        case_id=case_id,
        trial_count=len(ordered),
        passed_trials=passed_trials,
        execution_failures=execution_failures,
        assessment_errors=assessment_errors,
        passed=passed,
        metrics=metrics,
        trial_indices=tuple(trial.trial_index for trial in ordered),
        execution_statuses=statuses,
        missing_metric_names=missing,
    )


def _aggregate_metrics(trials: list[EvalTrial], policy: MetricAggregation) -> EvalMetrics:
    names = sorted({name for trial in trials for name in trial.metrics})
    aggregated: dict[str, EvalMetric] = {}
    for name in names:
        available = [trial.metrics.get(name) for trial in trials]
        present = [metric for metric in available if metric is not None]
        if not present:
            continue
        usable = [
            metric
            for metric in present
            if metric.value is not None and metric.completeness is not MetricCompleteness.MISSING
        ]
        template = present[0]
        if not usable:
            aggregated[name] = replace(
                template,
                value=None,
                completeness=MetricCompleteness.MISSING,
                scope=MetricScope.CASE,
                detail=f"all {len(trials)} trial(s) reported missing evidence",
            )
            continue
        values: list[int | float] = [metric.value for metric in usable if metric.value is not None]
        value = _select(values, policy)
        complete = len(usable) == len(trials) and all(
            metric.completeness is MetricCompleteness.COMPLETE for metric in usable
        )
        detail = template.detail
        if not complete:
            detail = (
                f"{len(usable)} of {len(trials)} trial(s) contributed; "
                "missing or partial evidence remains"
            )
        aggregated[name] = replace(
            template,
            value=value,
            scope=MetricScope.CASE,
            completeness=(MetricCompleteness.COMPLETE if complete else MetricCompleteness.PARTIAL),
            detail=detail,
        )
    return EvalMetrics(aggregated)


def _select(values: list[int | float], policy: MetricAggregation) -> int | float:
    if policy is MetricAggregation.WORST:
        return max(values)
    if policy is MetricAggregation.BEST:
        return min(values)
    if policy is MetricAggregation.MEAN:
        return sum(values) / len(values)
    return statistics.median(values)


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _expect_mapping(data.get(key), key)


def _expect_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _as_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must contain integers")
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
