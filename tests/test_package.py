import langgraph_instrumentation
from langgraph_instrumentation import stores


def test_public_api_is_explicit() -> None:
    assert langgraph_instrumentation.__all__ == [
        "Clock",
        "ClockReading",
        "CaptureMode",
        "CapturePolicy",
        "DeterministicClock",
        "DeterministicIdGenerator",
        "ExecutionLane",
        "FlushableTraceStore",
        "IdGenerator",
        "LangGraphCallbackHandler",
        "MessageProjector",
        "MemoryTraceStore",
        "MetadataProjector",
        "MetricPoint",
        "PerfettoExporter",
        "RandomIdGenerator",
        "RedactionPolicy",
        "RecorderError",
        "RecordingLimits",
        "RecordingSession",
        "Sanitizer",
        "Span",
        "SpanEvent",
        "SpanId",
        "SpanKind",
        "SpanLink",
        "SpanProcessor",
        "SpanStatus",
        "SQLiteTraceStore",
        "SystemClock",
        "TokenUsage",
        "Trace",
        "TraceBundle",
        "TraceId",
        "TraceQuery",
        "TraceRecorder",
        "TraceStore",
        "TraceSummary",
        "UsageExtractor",
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


def test_store_api_is_explicit() -> None:
    assert stores.__all__ == [
        "FlushableTraceStore",
        "MemoryTraceStore",
        "SQLiteTraceStore",
        "TraceStore",
    ]


def test_adapter_api_is_explicit() -> None:
    from langgraph_instrumentation import adapters

    assert adapters.__all__ == ["LangGraphCallbackHandler"]
