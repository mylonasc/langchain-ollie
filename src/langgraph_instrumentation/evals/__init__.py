"""Local-first, trace-native evaluation contracts and execution."""

from .adapters import (
    AsyncExecutionAdapter,
    EvalInvocation,
    ExecutionAdapter,
    LangGraphEvalAdapter,
)
from .datasets import load_cases, save_cases
from .evaluators import (
    ContainsText,
    EvalContext,
    Evaluator,
    ExactText,
    FunctionEvaluator,
    MaxMetric,
    MinMetric,
    RegexText,
    ToolCalled,
)
from .metrics import aggregate_turn_metrics, measure_traces
from .models import (
    EvalAssessment,
    EvalCase,
    EvalError,
    EvalExecutionStatus,
    EvalMetric,
    EvalMetrics,
    EvalOutput,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurn,
    EvalTurnResult,
    MetricCompleteness,
    MetricScope,
)
from .runner import EvalRunner

__all__ = [
    "AsyncExecutionAdapter",
    "ContainsText",
    "EvalAssessment",
    "EvalCase",
    "EvalContext",
    "EvalError",
    "EvalExecutionStatus",
    "EvalInvocation",
    "EvalMetric",
    "EvalMetrics",
    "EvalOutput",
    "EvalRun",
    "EvalRunner",
    "EvalStatus",
    "EvalTrial",
    "EvalTurn",
    "EvalTurnResult",
    "Evaluator",
    "ExactText",
    "ExecutionAdapter",
    "FunctionEvaluator",
    "LangGraphEvalAdapter",
    "MaxMetric",
    "MetricCompleteness",
    "MetricScope",
    "MinMetric",
    "RegexText",
    "ToolCalled",
    "aggregate_turn_metrics",
    "load_cases",
    "measure_traces",
    "save_cases",
]
