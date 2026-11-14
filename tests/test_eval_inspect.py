"""Offline tests for bounded agent-facing inspection operations."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from langgraph_instrumentation import (
    EvalAssessment,
    EvalError,
    EvalExecutionStatus,
    EvalMetrics,
    EvalOutput,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurnResult,
    RecordingSession,
    SpanKind,
)
from langgraph_instrumentation.evals.inspect import (
    summarize_failures,
    summarize_regression,
    turn_evidence,
)


def _turn(
    index: int,
    status: EvalExecutionStatus = EvalExecutionStatus.COMPLETED,
    assessments: tuple[EvalAssessment, ...] = (),
    output: EvalOutput | None = None,
    trace_ids: tuple[str, ...] = (),
) -> EvalTurnResult:
    return EvalTurnResult(
        index=index,
        input={"messages": [["user", "x"]]},
        execution_status=status,
        assessments=assessments,
        output=output,
        metrics=EvalMetrics(),
        trace_ids=trace_ids,
    )


def _trial(
    case_id: str,
    index: int,
    turns: tuple[EvalTurnResult, ...],
    status: EvalExecutionStatus = EvalExecutionStatus.COMPLETED,
    metrics: EvalMetrics | None = None,
    error: Any = None,
) -> EvalTrial:
    return EvalTrial(
        case_id=case_id,
        trial_index=index,
        execution_status=status,
        turns=turns,
        metrics=metrics or EvalMetrics(),
        error=error,
    )


def _run(trials: list[EvalTrial], run_id: str = "run") -> EvalRun:
    return EvalRun(run_id=run_id, trials=tuple(trials), started_unix_ns=0, finished_unix_ns=1)


def test_summarize_failures_is_bounded_and_marks_truncation() -> None:
    failing = _turn(
        0,
        assessments=(EvalAssessment("check", EvalStatus.FAIL, message="nope"),),
    )
    error = _turn(1, status=EvalExecutionStatus.AGENT_ERROR)
    trials = [
        _trial(
            "bad",
            0,
            (failing, error),
            EvalExecutionStatus.AGENT_ERROR,
            error=EvalError(exception_type="RuntimeError"),
        ),
        _trial("good", 0, (_turn(0),)),
        _trial("bad", 1, (failing,)),
    ]
    summary = summarize_failures(_run(trials), max_cases=1)
    assert summary.failed_trials == 2
    assert summary.total_trials == 3
    assert summary.truncated is True
    assert len(summary.failed_cases) == 1
    case = summary.failed_cases[0]
    assert case.case_id == "bad"
    assert case.error_type is not None
    assert any("check" in reason for reason in case.reasons)
    assert case.reasons_truncated is False

    full = summarize_failures(_run(trials))
    assert full.truncated is False
    assert len(full.failed_cases) == 2
    assert full.to_dict()["kind"] == "langgraph_instrumentation.failure_summary"


def test_summarize_failures_rejects_bad_bounds() -> None:
    run = _run([_trial("a", 0, (_turn(0),))])
    with pytest.raises(ValueError, match="max_cases"):
        summarize_failures(run, max_cases=0)


def _recorded_bundles() -> dict[str, Any]:
    session = RecordingSession()
    recorder = session.recorder
    llm_correlation, tool_correlation = uuid4().hex, uuid4().hex
    recorder.start_span(llm_correlation, "LLM: scripted", SpanKind.LLM)
    recorder.record_metric(
        llm_correlation,
        "llm.token.usage",
        {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    )
    recorder.end_span(llm_correlation)
    recorder.start_span(tool_correlation, "Tool: add", SpanKind.TOOL)
    recorder.end_span(tool_correlation)
    bundles = {str(bundle.trace.trace_id): bundle for bundle in session.snapshot()}
    assert session.close()
    assert len(bundles) == 2
    return bundles


def test_turn_evidence_reports_calls_and_missing_traces() -> None:
    bundles = _recorded_bundles()
    trace_ids = tuple(sorted(bundles))
    turn = _turn(
        0,
        assessments=(EvalAssessment("exact", EvalStatus.PASS),),
        output=EvalOutput(text="x" * 100),
        trace_ids=trace_ids,
    )
    trial = _trial("case", 0, (turn,))
    run = _run([trial])

    full = turn_evidence(run, "case", 0, traces=bundles)
    assert full.execution_status == "completed"
    assert full.missing_trace_ids == ()
    assert full.output_truncated is False
    assert {call.kind for call in full.calls} == {"llm", "tool"}
    assert full.calls_truncated is False
    assert full.to_dict()["kind"] == "langgraph_instrumentation.turn_evidence"

    partial = turn_evidence(run, "case", 0, traces={}, max_text=10, max_calls=1)
    assert partial.missing_trace_ids == trace_ids
    assert partial.calls == ()
    assert partial.output_truncated is True
    assert partial.output_text is not None and partial.output_text.endswith(">")

    limited = turn_evidence(run, "case", 0, traces=bundles, max_calls=1)
    assert len(limited.calls) == 1
    assert limited.calls_truncated is True


def test_turn_evidence_validates_inputs() -> None:
    bundles = _recorded_bundles()
    run = _run([_trial("case", 0, (_turn(0),))])
    with pytest.raises(KeyError, match="no turn 9"):
        turn_evidence(run, "case", 9, traces=bundles)
    with pytest.raises(KeyError, match="no turn"):
        turn_evidence(run, "absent", 0, traces=bundles)
    with pytest.raises(TypeError, match="must be an EvalRun"):
        turn_evidence(cast(Any, "nope"), "case", 0)
    with pytest.raises(TypeError, match="must be a TraceBundle"):
        turn_evidence(run, "case", 0, traces=cast(Any, {"x": "nope"}))


def test_summarize_regression_covers_outcomes_and_mismatches() -> None:
    from langgraph_instrumentation import MetricRule, compare_runs

    def trial(passed: bool, tokens: int) -> EvalTrial:
        status = EvalStatus.PASS if passed else EvalStatus.FAIL
        turn = _turn(0, assessments=(EvalAssessment("check", status),))
        return _trial("a", 0, (turn,), metrics=_tokens(tokens))

    baseline = _run([trial(True, 100)], run_id="base")
    candidate = _run([trial(False, 200)], run_id="cand")
    comparison = compare_runs(baseline, candidate, rules=(MetricRule("tokens.total"),))
    summary = summarize_regression(comparison)
    assert summary.passed is False
    assert summary.regressed_case_ids == ("a",)
    assert summary.cases[0].quality == "regressed"
    assert summary.cases[0].metric_deltas[0]["delta"] == 100.0
    assert summary.truncated is False
    assert summary.to_dict()["kind"] == "langgraph_instrumentation.regression_summary"


def _tokens(total: int) -> EvalMetrics:
    from langgraph_instrumentation import EvalMetric, MetricCompleteness, MetricScope

    return EvalMetrics(
        {
            "tokens.total": EvalMetric(
                name="tokens.total",
                value=total,
                unit="tokens",
                source="tests",
                scope=MetricScope.CASE,
                completeness=MetricCompleteness.COMPLETE,
            )
        }
    )
