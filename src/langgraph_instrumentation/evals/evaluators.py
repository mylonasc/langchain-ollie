"""Composable deterministic evaluators and explicit performance budgets."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..models import SpanKind, TraceBundle
from .models import (
    EvalAssessment,
    EvalCase,
    EvalMetrics,
    EvalOutput,
    EvalStatus,
    EvalTurn,
    EvalTurnResult,
    MetricCompleteness,
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
    missing: str = "fail"
    name: str = "max_metric"

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric name must not be empty")
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, (int, float)):
            raise TypeError("maximum must be a number")
        if self.missing not in ("fail", "error", "skip"):
            raise ValueError("missing must be 'fail', 'error' or 'skip'")

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        metric = context.metrics.get(self.metric)
        evidence: dict[str, object] = {
            "metric": self.metric,
            "maximum": self.maximum,
        }
        if metric is None or metric.completeness is MetricCompleteness.MISSING:
            status = {
                "fail": EvalStatus.FAIL,
                "error": EvalStatus.ERROR,
                "skip": EvalStatus.SKIP,
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
    missing: str = "fail"
    name: str = "min_metric"

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric name must not be empty")
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, (int, float)):
            raise TypeError("minimum must be a number")
        if self.missing not in ("fail", "error", "skip"):
            raise ValueError("missing must be 'fail', 'error' or 'skip'")

    def evaluate(self, context: EvalContext) -> EvalAssessment:
        metric = context.metrics.get(self.metric)
        evidence: dict[str, object] = {
            "metric": self.metric,
            "minimum": self.minimum,
        }
        if metric is None or metric.completeness is MetricCompleteness.MISSING:
            status = {
                "fail": EvalStatus.FAIL,
                "error": EvalStatus.ERROR,
                "skip": EvalStatus.SKIP,
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


def _output_text(context: EvalContext) -> str | None:
    return context.output.text if context.output is not None else None


def _tool_name_matches(observed: str, expected: str) -> bool:
    return (
        observed == expected
        or observed == f"Tool: {expected}"
        or observed.endswith(f": {expected}")
    )
