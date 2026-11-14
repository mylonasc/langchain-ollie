"""Neutral evaluation records shared by runners, evaluators and artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from ..models import (
    JSONValue,
    TraceBundle,
    attributes_to_dict,
    freeze_attributes,
)

_EMPTY_METADATA = MappingProxyType({})
_SCHEMA_VERSION = "1"
_RUN_KIND = "langgraph_instrumentation.eval_run"


class EvalStatus(StrEnum):
    """The outcome of one evaluator or check."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class EvalExecutionStatus(StrEnum):
    """The observed execution outcome, separate from assessment outcomes."""

    COMPLETED = "completed"
    AGENT_ERROR = "agent_error"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"
    RUNNER_ERROR = "runner_error"


class MetricCompleteness(StrEnum):
    """Whether a measurement is complete, partial or unavailable."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class MetricScope(StrEnum):
    """The entity a measurement describes."""

    TURN = "turn"
    TRIAL = "trial"
    CASE = "case"
    PROCESS = "process"


class MissingMetricPolicy(StrEnum):
    """How a check treats unavailable measurement evidence."""

    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class EvalError:
    """A bounded execution error without application payload by default."""

    exception_type: str
    message: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.exception_type, "exception_type")
        if self.message is not None:
            _validate_nonempty_string(self.message, "message")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"exception_type": self.exception_type, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalError:
        message = data.get("message")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be a string or null")
        return cls(
            exception_type=_required_string(data, "exception_type"),
            message=message,
        )


@dataclass(frozen=True, slots=True)
class EvalTurn:
    """One user interaction and its turn-level checks."""

    input: Mapping[str, JSONValue] | None = None
    messages: tuple[tuple[str, str], ...] = ()
    checks: tuple[object, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)
    tags: frozenset[str] = frozenset()
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if (self.input is None) == (not self.messages):
            raise ValueError("a turn must define exactly one of input or messages")
        if self.input is not None:
            _validate_json_mapping(self.input, "turn input")
        messages = tuple((role, content) for role, content in self.messages)
        for role, content in messages:
            _validate_nonempty_string(role, "message role")
            if not isinstance(content, str):
                raise TypeError("message content must be a string")
        checks = tuple(self.checks)
        for check in checks:
            if not hasattr(check, "evaluate") or not hasattr(check, "name"):
                raise TypeError("turn checks must implement the Evaluator protocol")
        object.__setattr__(self, "messages", messages)
        if self.input is not None:
            object.__setattr__(self, "input", freeze_attributes(self.input))
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))
        object.__setattr__(self, "tags", frozenset(self.tags))
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise TypeError("timeout_seconds must be a number or null")
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive and finite")

    @classmethod
    def user(
        cls,
        content: str,
        *,
        checks: tuple[object, ...] = (),
        metadata: Mapping[str, object] | None = None,
        tags: frozenset[str] = frozenset(),
        timeout_seconds: float | None = None,
    ) -> EvalTurn:
        """Build a single-user-message turn."""
        return cls(
            messages=(("user", content),),
            checks=checks,
            metadata=metadata or {},
            tags=tags,
            timeout_seconds=timeout_seconds,
        )

    def payload(self) -> Mapping[str, JSONValue]:
        """Return the adapter-neutral invocation payload."""
        if self.input is not None:
            return cast(Mapping[str, JSONValue], attributes_to_dict(self.input))
        return {"messages": [[role, content] for role, content in self.messages]}

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "input": (None if self.input is None else attributes_to_dict(self.input)),
                "messages": [[role, content] for role, content in self.messages],
                "checks": [str(getattr(check, "name", "custom")) for check in self.checks],
                "metadata": attributes_to_dict(self.metadata),
                "tags": sorted(self.tags),
                "timeout_seconds": self.timeout_seconds,
            },
        )


@dataclass(frozen=True, slots=True)
class EvalCase:
    """An immutable evaluation scenario with one or more turns."""

    case_id: str
    turns: tuple[EvalTurn, ...]
    checks: tuple[object, ...] = ()
    name: str | None = None
    description: str | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, "case_id")
        turns = tuple(self.turns)
        if not turns:
            raise ValueError("an evaluation case must have at least one turn")
        for turn in turns:
            if not isinstance(turn, EvalTurn):
                raise TypeError("case turns must be EvalTurn records")
        checks = tuple(self.checks)
        for check in checks:
            if not hasattr(check, "evaluate") or not hasattr(check, "name"):
                raise TypeError("case checks must implement the Evaluator protocol")
        if self.name is not None:
            _validate_nonempty_string(self.name, "case name")
        if self.description is not None:
            _validate_nonempty_string(self.description, "case description")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))
        object.__setattr__(self, "tags", frozenset(self.tags))

    @classmethod
    def qa(
        cls,
        case_id: str,
        question: str,
        *,
        checks: tuple[object, ...] = (),
        name: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, object] | None = None,
        tags: frozenset[str] = frozenset(),
    ) -> EvalCase:
        """Build a single-turn question/answer case."""
        return cls(
            case_id=case_id,
            turns=(EvalTurn.user(question),),
            checks=checks,
            name=name,
            description=description,
            metadata=metadata or {},
            tags=tags,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            {
                "case_id": self.case_id,
                "name": self.name,
                "description": self.description,
                "turns": [turn.to_dict() for turn in self.turns],
                "checks": [str(getattr(check, "name", "custom")) for check in self.checks],
                "metadata": attributes_to_dict(self.metadata),
                "tags": sorted(self.tags),
            },
        )


@dataclass(frozen=True, slots=True)
class EvalOutput:
    """A JSON-safe agent output plus an optional stable text projection."""

    value: JSONValue = None
    text: str | None = None

    def __post_init__(self) -> None:
        _validate_json_value(self.value, "output")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("output text must be a string or null")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"value": self.value, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalOutput:
        text = data.get("text")
        if text is not None and not isinstance(text, str):
            raise TypeError("output text must be a string or null")
        return cls(value=cast(JSONValue, data.get("value")), text=text)


@dataclass(frozen=True, slots=True)
class EvalMetric:
    """One measurement with explicit source, unit, scope and completeness."""

    name: str
    value: int | float | None
    unit: str
    source: str
    scope: MetricScope
    completeness: MetricCompleteness = MetricCompleteness.COMPLETE
    detail: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.name, "metric name")
        _validate_nonempty_string(self.unit, "metric unit")
        _validate_nonempty_string(self.source, "metric source")
        if not isinstance(self.scope, MetricScope):
            raise TypeError("metric scope must be a MetricScope")
        if not isinstance(self.completeness, MetricCompleteness):
            raise TypeError("metric completeness must be a MetricCompleteness")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError("metric value must be a number or null")
            if not math.isfinite(self.value):
                raise ValueError("metric value must be finite")
        if self.completeness is MetricCompleteness.MISSING and self.value is not None:
            raise ValueError("a missing metric cannot have a value")
        if self.completeness is not MetricCompleteness.MISSING and self.value is None:
            raise ValueError("a present metric must have a value")
        if self.detail is not None:
            _validate_nonempty_string(self.detail, "metric detail")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "scope": self.scope.value,
            "completeness": self.completeness.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalMetric:
        return cls(
            name=_required_string(data, "name"),
            value=_optional_number(data, "value"),
            unit=_required_string(data, "unit"),
            source=_required_string(data, "source"),
            scope=MetricScope(_required_string(data, "scope")),
            completeness=MetricCompleteness(_required_string(data, "completeness")),
            detail=_optional_string(data, "detail"),
        )


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """A deterministic collection of named measurements."""

    values: Mapping[str, EvalMetric] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        values = dict(self.values)
        for name, metric in values.items():
            if not isinstance(metric, EvalMetric):
                raise TypeError("metric values must be EvalMetric records")
            if name != metric.name:
                raise ValueError("metric mapping keys must match metric names")
        object.__setattr__(self, "values", MappingProxyType(dict(sorted(values.items()))))

    def get(self, name: str) -> EvalMetric | None:
        return self.values.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.values

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, JSONValue]:
        return {name: metric.to_dict() for name, metric in self.values.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalMetrics:
        return cls(
            {
                name: EvalMetric.from_dict(_expect_mapping(value, f"metric {name}"))
                for name, value in data.items()
            }
        )


@dataclass(frozen=True, slots=True)
class EvalAssessment:
    """A judgment with its status, evidence and optional score."""

    evaluator: str
    status: EvalStatus
    message: str | None = None
    score: float | None = None
    evidence: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.evaluator, "evaluator name")
        if not isinstance(self.status, EvalStatus):
            raise TypeError("assessment status must be an EvalStatus")
        if self.message is not None:
            _validate_nonempty_string(self.message, "assessment message")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError("assessment score must be a number or null")
            if not math.isfinite(self.score):
                raise ValueError("assessment score must be finite")
        object.__setattr__(self, "evidence", freeze_attributes(self.evidence))

    @property
    def passed(self) -> bool:
        return self.status in (EvalStatus.PASS, EvalStatus.SKIP)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "evaluator": self.evaluator,
            "status": self.status.value,
            "message": self.message,
            "score": self.score,
            "evidence": attributes_to_dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalAssessment:
        return cls(
            evaluator=_required_string(data, "evaluator"),
            status=EvalStatus(_required_string(data, "status")),
            message=_optional_string(data, "message"),
            score=_optional_number(data, "score"),
            evidence=_attributes_from_data(data, "evidence"),
        )


@dataclass(frozen=True, slots=True)
class EvalTurnResult:
    """Observed execution, measurements and assessments for one turn."""

    index: int
    input: Mapping[str, JSONValue]
    execution_status: EvalExecutionStatus
    assessments: tuple[EvalAssessment, ...] = ()
    output: EvalOutput | None = None
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    trace_ids: tuple[str, ...] = ()
    error: EvalError | None = None
    started_unix_ns: int = 0
    finished_unix_ns: int = 0
    duration_ns: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("turn index must be a non-negative integer")
        _validate_json_mapping(self.input, "turn result input")
        if not isinstance(self.execution_status, EvalExecutionStatus):
            raise TypeError("execution_status must be an EvalExecutionStatus")
        assessments = tuple(self.assessments)
        for assessment in assessments:
            if not isinstance(assessment, EvalAssessment):
                raise TypeError("assessments must be EvalAssessment records")
        if self.output is not None and not isinstance(self.output, EvalOutput):
            raise TypeError("output must be EvalOutput or null")
        if not isinstance(self.metrics, EvalMetrics):
            raise TypeError("metrics must be EvalMetrics")
        trace_ids = tuple(self.trace_ids)
        for trace_id in trace_ids:
            _validate_nonempty_string(trace_id, "trace ID")
        if self.error is not None and not isinstance(self.error, EvalError):
            raise TypeError("error must be EvalError or null")
        for name, value in (
            ("started_unix_ns", self.started_unix_ns),
            ("finished_unix_ns", self.finished_unix_ns),
            ("duration_ns", self.duration_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "input", freeze_attributes(self.input))
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "trace_ids", trace_ids)

    @property
    def passed(self) -> bool:
        return self.execution_status is EvalExecutionStatus.COMPLETED and all(
            assessment.passed for assessment in self.assessments
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "index": self.index,
            "input": attributes_to_dict(self.input),
            "execution_status": self.execution_status.value,
            "output": self.output.to_dict() if self.output else None,
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "metrics": self.metrics.to_dict(),
            "trace_ids": list(self.trace_ids),
            "error": self.error.to_dict() if self.error else None,
            "started_unix_ns": self.started_unix_ns,
            "finished_unix_ns": self.finished_unix_ns,
            "duration_ns": self.duration_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalTurnResult:
        output = data.get("output")
        error = data.get("error")
        return cls(
            index=int(_required_integer(data, "index")),
            input=_json_mapping(data, "input"),
            execution_status=EvalExecutionStatus(_required_string(data, "execution_status")),
            output=(
                None if output is None else EvalOutput.from_dict(_expect_mapping(output, "output"))
            ),
            assessments=tuple(
                EvalAssessment.from_dict(_expect_mapping(item, "assessment"))
                for item in _required_list(data, "assessments", default=[])
            ),
            metrics=EvalMetrics.from_dict(_expect_mapping(data.get("metrics"), "metrics")),
            trace_ids=tuple(str(item) for item in _required_list(data, "trace_ids", default=[])),
            error=None if error is None else EvalError.from_dict(_expect_mapping(error, "error")),
            started_unix_ns=_required_integer(data, "started_unix_ns"),
            finished_unix_ns=_required_integer(data, "finished_unix_ns"),
            duration_ns=_required_integer(data, "duration_ns"),
        )


@dataclass(frozen=True, slots=True)
class EvalTrial:
    """One observed execution of a case across all of its turns."""

    case_id: str
    trial_index: int
    execution_status: EvalExecutionStatus
    turns: tuple[EvalTurnResult, ...]
    assessments: tuple[EvalAssessment, ...] = ()
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    error: EvalError | None = None
    started_unix_ns: int = 0
    finished_unix_ns: int = 0
    duration_ns: int = 0
    metadata: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, "case_id")
        if isinstance(self.trial_index, bool) or not isinstance(self.trial_index, int):
            raise TypeError("trial_index must be an integer")
        if self.trial_index < 0:
            raise ValueError("trial_index must be non-negative")
        if not isinstance(self.execution_status, EvalExecutionStatus):
            raise TypeError("execution_status must be an EvalExecutionStatus")
        turns = tuple(self.turns)
        for turn in turns:
            if not isinstance(turn, EvalTurnResult):
                raise TypeError("turns must be EvalTurnResult records")
        assessments = tuple(self.assessments)
        for assessment in assessments:
            if not isinstance(assessment, EvalAssessment):
                raise TypeError("assessments must be EvalAssessment records")
        if not isinstance(self.metrics, EvalMetrics):
            raise TypeError("metrics must be EvalMetrics")
        if self.error is not None and not isinstance(self.error, EvalError):
            raise TypeError("error must be EvalError or null")
        for name, value in (
            ("started_unix_ns", self.started_unix_ns),
            ("finished_unix_ns", self.finished_unix_ns),
            ("duration_ns", self.duration_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(trace_id for turn in self.turns for trace_id in turn.trace_ids)

    @property
    def passed(self) -> bool:
        return (
            self.execution_status is EvalExecutionStatus.COMPLETED
            and all(turn.passed for turn in self.turns)
            and all(assessment.passed for assessment in self.assessments)
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "trial_index": self.trial_index,
            "execution_status": self.execution_status.value,
            "turns": [turn.to_dict() for turn in self.turns],
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "metrics": self.metrics.to_dict(),
            "error": self.error.to_dict() if self.error else None,
            "started_unix_ns": self.started_unix_ns,
            "finished_unix_ns": self.finished_unix_ns,
            "duration_ns": self.duration_ns,
            "metadata": attributes_to_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalTrial:
        error = data.get("error")
        return cls(
            case_id=_required_string(data, "case_id"),
            trial_index=int(_required_integer(data, "trial_index")),
            execution_status=EvalExecutionStatus(_required_string(data, "execution_status")),
            turns=tuple(
                EvalTurnResult.from_dict(_expect_mapping(item, "turn"))
                for item in _required_list(data, "turns", default=[])
            ),
            assessments=tuple(
                EvalAssessment.from_dict(_expect_mapping(item, "assessment"))
                for item in _required_list(data, "assessments", default=[])
            ),
            metrics=EvalMetrics.from_dict(_expect_mapping(data.get("metrics"), "metrics")),
            error=None if error is None else EvalError.from_dict(_expect_mapping(error, "error")),
            started_unix_ns=_required_integer(data, "started_unix_ns"),
            finished_unix_ns=_required_integer(data, "finished_unix_ns"),
            duration_ns=_required_integer(data, "duration_ns"),
            metadata=_attributes_from_data(data, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class EvalRun:
    """A completed collection of trials with shared reproducibility metadata."""

    run_id: str
    trials: tuple[EvalTrial, ...]
    started_unix_ns: int
    finished_unix_ns: int
    metadata: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)
    environment: Mapping[str, object] = field(default_factory=lambda: _EMPTY_METADATA)
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema version: {self.schema_version!r}")
        trials = tuple(self.trials)
        for trial in trials:
            if not isinstance(trial, EvalTrial):
                raise TypeError("trials must be EvalTrial records")
        for name, value in (
            ("started_unix_ns", self.started_unix_ns),
            ("finished_unix_ns", self.finished_unix_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.finished_unix_ns < self.started_unix_ns:
            raise ValueError("run finish cannot precede its start")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "metadata", freeze_attributes(self.metadata))
        object.__setattr__(self, "environment", freeze_attributes(self.environment))

    @property
    def passed(self) -> bool:
        return all(trial.passed for trial in self.trials)

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(trace_id for trial in self.trials for trace_id in trial.trace_ids)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _RUN_KIND,
            "run_id": self.run_id,
            "started_unix_ns": self.started_unix_ns,
            "finished_unix_ns": self.finished_unix_ns,
            "metadata": attributes_to_dict(self.metadata),
            "environment": attributes_to_dict(self.environment),
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalRun:
        kind = data.get("kind", _RUN_KIND)
        if kind != _RUN_KIND:
            raise ValueError(f"not an evaluation run artifact: {kind!r}")
        version = _required_string(data, "schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema version: {version!r}")
        return cls(
            run_id=_required_string(data, "run_id"),
            trials=tuple(
                EvalTrial.from_dict(_expect_mapping(item, "trial"))
                for item in _required_list(data, "trials", default=[])
            ),
            started_unix_ns=_required_integer(data, "started_unix_ns"),
            finished_unix_ns=_required_integer(data, "finished_unix_ns"),
            metadata=_attributes_from_data(data, "metadata"),
            environment=_attributes_from_data(data, "environment"),
            schema_version=version,
        )


def trace_bundle_ids(bundles: tuple[TraceBundle, ...]) -> tuple[str, ...]:
    """Return stable trace references for evaluation artifacts."""
    return tuple(str(bundle.trace.trace_id) for bundle in bundles)


def _validate_json_mapping(value: Mapping[str, object], field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    _validate_json_value(dict(value), field_name)


def _validate_json_value(value: object, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON-compatible") from exc


def _validate_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_identifier(value: object, field_name: str) -> None:
    _validate_nonempty_string(value, field_name)
    assert isinstance(value, str)
    if len(value) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain whitespace or control characters")


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


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_number(data: Mapping[str, object], key: str) -> int | float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number or null")
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


def _expect_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _json_mapping(data: Mapping[str, object], key: str) -> Mapping[str, JSONValue]:
    value = _expect_mapping(data.get(key), key)
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{key} keys must be strings")
    _validate_json_value(dict(value), key)
    return cast(Mapping[str, JSONValue], value)


def _attributes_from_data(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if value is None:
        return _EMPTY_METADATA
    return freeze_attributes(_expect_mapping(value, key))
