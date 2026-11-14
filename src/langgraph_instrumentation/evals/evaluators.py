"""Composable deterministic evaluators and explicit performance budgets."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..models import JSONValue, SpanKind, TraceBundle, freeze_attributes
from .models import (
    EvalAssessment,
    EvalCase,
    EvalMetrics,
    EvalOutput,
    EvalStatus,
    EvalTurn,
    EvalTurnResult,
    MetricCompleteness,
    MissingMetricPolicy,
)


@dataclass(frozen=True, slots=True)
class EvalContext:
    """The bounded view supplied to an evaluator."""

    case: EvalCase
    trial_index: int
    turn: EvalTurn | None
    turn_index: int | None
    output: EvalOutput | None
    metrics: EvalMetrics
    traces: tuple[TraceBundle, ...] = ()
    prior_turns: tuple[EvalTurnResult, ...] = ()

    def __post_init__(self) -> None:
        if (self.turn is None) != (self.turn_index is None):
            raise ValueError("turn and turn_index must both be set or both be absent")
        if not isinstance(self.metrics, EvalMetrics):
            raise TypeError("metrics must be EvalMetrics")
        if not isinstance(self.traces, tuple):
            raise TypeError("traces must be a tuple")
        if not isinstance(self.prior_turns, tuple):
            raise TypeError("prior_turns must be a tuple")

    @property
    def is_conversation_check(self) -> bool:
        return self.turn is None


@runtime_checkable
class Evaluator(Protocol):
    """Apply a check to a turn or conversation view."""

    @property
    def name(self) -> str:
        """Stable evaluator identity for results and comparisons."""
        ...

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        """Return one assessment with evidence."""
        ...


@dataclass(frozen=True, slots=True)
class FunctionEvaluator:
    """Wrap an application-defined callable."""

    name: str
    function: Callable[[EvalContext], bool | EvalAssessment]
    description: str | None = None

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        try:
            result = self.function(context)
        except Exception as exc:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message=f"evaluator raised {type(exc).__name__}",
            )
        if isinstance(result, EvalAssessment):
            return result
        if not isinstance(result, bool):
            raise TypeError("an evaluator function must return bool or EvalAssessment")
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if result else EvalStatus.FAIL,
        )


@dataclass(frozen=True, slots=True)
class ExactText:
    """Require the adapter's output text to equal an expected value."""

    expected: str
    case_sensitive: bool = True
    strip: bool = False
    name: str = "exact_text"

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        observed = _output_text(context)
        if observed is None:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message="the adapter did not provide output text",
                evidence={"expected": self.expected},
            )
        left, right = (
            (observed, self.expected)
            if self.case_sensitive
            else (observed.casefold(), self.expected.casefold())
        )
        if self.strip:
            left, right = left.strip(), right.strip()
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if left == right else EvalStatus.FAIL,
            message=None if left == right else "output text did not match",
            evidence={"observed": observed, "expected": self.expected},
        )


@dataclass(frozen=True, slots=True)
class ContainsText:
    """Require the output text to contain a fragment."""

    expected: str
    case_sensitive: bool = True
    name: str = "contains_text"

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        observed = _output_text(context)
        if observed is None:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message="the adapter did not provide output text",
                evidence={"expected": self.expected},
            )
        left, right = (
            (observed, self.expected)
            if self.case_sensitive
            else (observed.casefold(), self.expected.casefold())
        )
        passed = right in left
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            message=None if passed else "output text did not contain the expected fragment",
            evidence={"observed": observed, "expected": self.expected},
        )


@dataclass(frozen=True, slots=True)
class RegexText:
    """Require the output text to match a regular expression."""

    pattern: str
    flags: int = 0
    name: str = "regex_text"

    def __post_init__(self) -> None:
        re.compile(self.pattern, self.flags)

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        observed = _output_text(context)
        if observed is None:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message="the adapter did not provide output text",
                evidence={"pattern": self.pattern},
            )
        match = re.search(self.pattern, observed, self.flags)
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if match else EvalStatus.FAIL,
            message=None if match else "output text did not match the pattern",
            evidence={
                "observed": observed,
                "pattern": self.pattern,
                "match": match.group(0) if match else None,
            },
        )


@dataclass(frozen=True, slots=True)
class ToolCalled:
    """Require at least one recorded tool call with a given name."""

    tool: str
    minimum: int = 1
    name: str = "tool_called"

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("tool name must not be empty")
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise TypeError("minimum must be an integer")
        if self.minimum < 1:
            raise ValueError("minimum must be at least one")

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        names = [
            span.name
            for bundle in context.traces
            for span in bundle.spans
            if span.kind is SpanKind.TOOL
        ]
        matches = [name for name in names if _tool_name_matches(name, self.tool)]
        passed = len(matches) >= self.minimum
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            message=(
                None if passed else f"expected at least {self.minimum} call(s) to {self.tool!r}"
            ),
            evidence={
                "tool": self.tool,
                "observed": names,
                "matching": matches,
                "minimum": self.minimum,
            },
        )


@dataclass(frozen=True, slots=True)
class MaxMetric:
    """Fail when a named metric exceeds a maximum.

    Missing metrics have explicit policy: ``fail``, ``error`` or ``skip``.
    """

    metric: str
    maximum: int | float
    missing: MissingMetricPolicy = MissingMetricPolicy.FAIL
    name: str = "max_metric"

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric name must not be empty")
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, (int, float)):
            raise TypeError("maximum must be a number")
        try:
            policy = MissingMetricPolicy(self.missing)
        except ValueError as exc:
            raise ValueError("missing must be 'fail', 'error' or 'skip'") from exc
        object.__setattr__(self, "missing", policy)

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        metric = context.metrics.get(self.metric)
        evidence: dict[str, object] = {
            "metric": self.metric,
            "maximum": self.maximum,
            "missing_policy": self.missing.value,
        }
        if metric is None or metric.completeness is MetricCompleteness.MISSING:
            status = {
                MissingMetricPolicy.FAIL: EvalStatus.FAIL,
                MissingMetricPolicy.ERROR: EvalStatus.ERROR,
                MissingMetricPolicy.SKIP: EvalStatus.SKIP,
            }[self.missing]
            return EvalAssessment(
                evaluator=self.name,
                status=status,
                message=f"metric {self.metric!r} is missing",
                evidence=evidence,
            )
        assert metric.value is not None
        evidence.update(
            {
                "observed": metric.value,
                "unit": metric.unit,
                "source": metric.source,
                "scope": metric.scope.value,
                "completeness": metric.completeness.value,
                "detail": metric.detail,
            }
        )
        passed = metric.value <= self.maximum
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            message=(
                None if passed else f"{self.metric} exceeded {self.maximum} {metric.unit}".strip()
            ),
            evidence=evidence,
        )


@dataclass(frozen=True, slots=True)
class MinMetric:
    """Fail when a named metric is below a minimum."""

    metric: str
    minimum: int | float
    missing: MissingMetricPolicy = MissingMetricPolicy.FAIL
    name: str = "min_metric"

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric name must not be empty")
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, (int, float)):
            raise TypeError("minimum must be a number")
        try:
            policy = MissingMetricPolicy(self.missing)
        except ValueError as exc:
            raise ValueError("missing must be 'fail', 'error' or 'skip'") from exc
        object.__setattr__(self, "missing", policy)

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        metric = context.metrics.get(self.metric)
        evidence: dict[str, object] = {
            "metric": self.metric,
            "minimum": self.minimum,
            "missing_policy": self.missing.value,
        }
        if metric is None or metric.completeness is MetricCompleteness.MISSING:
            status = {
                MissingMetricPolicy.FAIL: EvalStatus.FAIL,
                MissingMetricPolicy.ERROR: EvalStatus.ERROR,
                MissingMetricPolicy.SKIP: EvalStatus.SKIP,
            }[self.missing]
            return EvalAssessment(
                evaluator=self.name,
                status=status,
                message=f"metric {self.metric!r} is missing",
                evidence=evidence,
            )
        assert metric.value is not None
        evidence.update(
            {
                "observed": metric.value,
                "unit": metric.unit,
                "source": metric.source,
                "scope": metric.scope.value,
                "completeness": metric.completeness.value,
                "detail": metric.detail,
            }
        )
        passed = metric.value >= self.minimum
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            message=(
                None if passed else f"{self.metric} was below {self.minimum} {metric.unit}".strip()
            ),
            evidence=evidence,
        )


@dataclass(frozen=True, slots=True)
class StatePredicate:
    """Require a dotted path in the structured output to equal a value.

    The path walks JSON objects by key and lists by numeric index, for example
    ``result.count`` or ``items.0.status``. This checks structured state without
    making the evaluator depend on an application class.
    """

    path: str
    expected: JSONValue
    name: str = "state_predicate"

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("path must not be empty")

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        if context.output is None:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message="the adapter did not provide structured output",
                evidence={"path": self.path, "expected": self.expected},
            )
        found, observed = _lookup_path(context.output.value, self.path)
        passed = found and _json_equal(observed, self.expected)
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if passed else EvalStatus.FAIL,
            message=(
                None if passed else f"no value at {self.path!r} matched the expected JSON value"
            ),
            evidence={
                "path": self.path,
                "expected": self.expected,
                "observed": observed,
                "found": found,
            },
        )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A structured judge decision; the judge itself is caller-supplied."""

    passed: bool
    reasoning: str | None = None
    score: float | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        if self.reasoning is not None and (
            not isinstance(self.reasoning, str) or not self.reasoning.strip()
        ):
            raise ValueError("reasoning must be a non-empty string or null")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError("score must be a number or null")
            if self.score != self.score or self.score in (float("inf"), float("-inf")):
                raise ValueError("score must be finite")
        object.__setattr__(self, "evidence", freeze_attributes(self.evidence))


@runtime_checkable
class Judge(Protocol):
    """Optional quality judge, typically backed by an application or adapter."""

    @property
    def name(self) -> str:
        """Stable judge identity for results."""
        ...

    def judge(self, context: EvalContext, /) -> JudgeVerdict:
        """Return a structured verdict; provider failures raise."""
        ...


@dataclass(frozen=True, slots=True)
class JudgeEvaluator:
    """Adapt a ``Judge`` to the deterministic evaluator protocol.

    The core package imports no provider SDK and never requires a judge. The
    caller supplies a judge explicitly; exceptions become assessment errors.
    """

    judge: Judge
    override_name: str | None = None

    @property
    def name(self) -> str:
        return self.override_name or self.judge.name

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        try:
            verdict = self.judge.judge(context)
        except Exception as exc:
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message=f"judge raised {type(exc).__name__}",
            )
        if not isinstance(verdict, JudgeVerdict):
            return EvalAssessment(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                message="judge did not return a JudgeVerdict",
            )
        evidence = {"judge": self.judge.name}
        evidence.update(dict(verdict.evidence))
        return EvalAssessment(
            evaluator=self.name,
            status=EvalStatus.PASS if verdict.passed else EvalStatus.FAIL,
            message=verdict.reasoning,
            score=verdict.score,
            evidence=evidence,
        )


def _output_text(context: EvalContext) -> str | None:
    return context.output.text if context.output is not None else None


def _tool_name_matches(observed: str, expected: str) -> bool:
    return (
        observed == expected
        or observed == f"Tool: {expected}"
        or observed.endswith(f": {expected}")
    )


def _lookup_path(value: JSONValue, path: str) -> tuple[bool, JSONValue]:
    current: JSONValue = value
    for segment in path.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                return False, None
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return False, None
            if not 0 <= index < len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def _json_equal(observed: JSONValue, expected: JSONValue) -> bool:
    if isinstance(observed, bool) != isinstance(expected, bool):
        return False
    return observed == expected
