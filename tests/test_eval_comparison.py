"""Offline tests for trial aggregation, comparisons, judges and state checks."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_ollie import (
    CaseAggregate,
    CaseComparison,
    ComparisonKind,
    ComparisonSummary,
    EvalAssessment,
    EvalCase,
    EvalContext,
    EvalExecutionStatus,
    EvalMetric,
    EvalMetrics,
    EvalOutput,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurnResult,
    JudgeEvaluator,
    JudgeVerdict,
    MetricAggregation,
    MetricCompleteness,
    MetricDirection,
    MetricRule,
    MetricScope,
    MissingMetricPolicy,
    RunAggregate,
    StatePredicate,
    TrialAggregation,
    aggregate_run,
    compare_runs,
)


def metric(
    name: str,
    value: int | float | None,
    *,
    completeness: MetricCompleteness = MetricCompleteness.COMPLETE,
    unit: str = "tokens",
) -> EvalMetric:
    return EvalMetric(
        name=name,
        value=value,
        unit=unit,
        source="tests",
        scope=MetricScope.CASE,
        completeness=completeness,
    )


def trial(
    case_id: str,
    *,
    index: int = 0,
    passed: bool = True,
    status: EvalExecutionStatus = EvalExecutionStatus.COMPLETED,
    metrics: EvalMetrics | None = None,
    assessment_error: bool = False,
) -> EvalTrial:
    assessments: tuple[EvalAssessment, ...] = ()
    if assessment_error:
        assessments = (EvalAssessment("check", EvalStatus.ERROR),)
    elif not passed and status is EvalExecutionStatus.COMPLETED:
        assessments = (EvalAssessment("check", EvalStatus.FAIL),)
    turn = EvalTurnResult(
        index=0,
        input={"messages": [["user", "x"]]},
        execution_status=status,
        assessments=assessments,
        metrics=metrics or EvalMetrics(),
    )
    return EvalTrial(
        case_id=case_id,
        trial_index=index,
        execution_status=status,
        turns=(turn,),
        metrics=metrics or EvalMetrics(),
    )


def run(
    run_id: str,
    trials: list[EvalTrial],
    *,
    metadata: dict[str, Any] | None = None,
) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        trials=tuple(trials),
        started_unix_ns=0,
        finished_unix_ns=1,
        metadata=metadata or {},
    )


def case_of(aggregate: RunAggregate, case_id: str) -> CaseAggregate:
    case = aggregate.case(case_id)
    assert case is not None
    return case


def compared(summary: ComparisonSummary, case_id: str) -> CaseComparison:
    comparison = summary.case(case_id)
    assert comparison is not None
    return comparison


def test_worst_trial_aggregation_is_fail_safe() -> None:
    mixed = run("r", [trial("a", index=0), trial("a", index=1, passed=False)])
    worst = aggregate_run(mixed)
    case = worst.case("a")
    assert case is not None
    assert case.trial_count == 2
    assert case.passed_trials == 1
    assert case.pass_rate == 0.5
    assert not case.passed
    assert not worst.passed

    assert case_of(aggregate_run(mixed, trials=TrialAggregation.BEST), "a").passed
    assert not case_of(aggregate_run(mixed, trials=TrialAggregation.MAJORITY), "a").passed

    mostly = run("r", [trial("a", index=0), trial("a", index=1), trial("a", index=2, passed=False)])
    assert case_of(aggregate_run(mostly, trials=TrialAggregation.MAJORITY), "a").passed


def test_metric_aggregation_modes_and_missing_evidence() -> None:
    ten = EvalMetrics({"tokens.total": metric("tokens.total", 10)})
    twenty = EvalMetrics({"tokens.total": metric("tokens.total", 20)})
    sample = run("r", [trial("a", index=0, metrics=ten), trial("a", index=1, metrics=twenty)])

    def value(policy: MetricAggregation) -> Any:
        case = case_of(aggregate_run(sample, metrics=policy), "a")
        aggregated = case.metrics.get("tokens.total")
        assert aggregated is not None
        return aggregated.value

    assert value(MetricAggregation.WORST) == 20
    assert value(MetricAggregation.BEST) == 10
    assert value(MetricAggregation.MEAN) == 15
    assert value(MetricAggregation.MEDIAN) == 15

    missing = EvalMetrics(
        {"tokens.total": metric("tokens.total", None, completeness=MetricCompleteness.MISSING)}
    )
    partial = run(
        "r",
        [trial("a", index=0, metrics=ten), trial("a", index=1, metrics=missing)],
    )
    case = case_of(aggregate_run(partial), "a")
    aggregated = case.metrics.get("tokens.total")
    assert aggregated is not None
    assert aggregated.value == 10
    assert aggregated.completeness is MetricCompleteness.PARTIAL
    assert "1 of 2" in (aggregated.detail or "")

    absent = run("r", [trial("a", index=0, metrics=missing), trial("a", index=1, metrics=missing)])
    absent_case = case_of(aggregate_run(absent), "a")
    absent_metric = absent_case.metrics.get("tokens.total")
    assert absent_metric is not None
    assert absent_metric.value is None
    assert absent_case.missing_metric_names == ("tokens.total",)


def test_comparison_separates_quality_and_performance() -> None:
    def metrics(tokens: int) -> EvalMetrics:
        return EvalMetrics({"tokens.total": metric("tokens.total", tokens)})

    baseline = run(
        "base",
        [trial("same", metrics=metrics(100)), trial("breaks", metrics=metrics(100))],
    )
    candidate = run(
        "cand",
        [trial("same", metrics=metrics(80)), trial("breaks", passed=False, metrics=metrics(100))],
    )
    summary = compare_runs(
        baseline,
        candidate,
        rules=(MetricRule("tokens.total", MetricDirection.LOWER_IS_BETTER),),
    )

    improved = compared(summary, "same")
    assert improved.kind is ComparisonKind.IMPROVED
    assert improved.quality is ComparisonKind.UNCHANGED
    assert improved.metrics[0].delta == -20.0

    regressed = compared(summary, "breaks")
    assert regressed.kind is ComparisonKind.REGRESSED
    assert regressed.quality is ComparisonKind.REGRESSED
    assert "case quality regressed" in regressed.reasons
    assert not summary.passed
    assert summary.improved_case_ids == ("same",)
    assert summary.regressed_case_ids == ("breaks",)


def test_metric_tolerance_and_direction() -> None:
    baseline = run("base", [trial("a", metrics=EvalMetrics({"x": metric("x", 100, unit="ms")}))])
    candidate = run("cand", [trial("a", metrics=EvalMetrics({"x": metric("x", 105, unit="ms")}))])

    tolerated = compare_runs(
        baseline,
        candidate,
        rules=(MetricRule("x", relative_tolerance=0.10),),
    )
    assert compared(tolerated, "a").kind is ComparisonKind.UNCHANGED

    strict = compare_runs(baseline, candidate, rules=(MetricRule("x"),))
    strict_case = compared(strict, "a")
    assert strict_case.kind is ComparisonKind.REGRESSED
    assert strict_case.metrics[0].relative_delta == pytest.approx(0.05)

    higher = compare_runs(
        baseline,
        candidate,
        rules=(
            MetricRule(
                "x",
                direction=MetricDirection.HIGHER_IS_BETTER,
                absolute_tolerance=4,
            ),
        ),
    )
    assert compared(higher, "a").kind is ComparisonKind.IMPROVED


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (MissingMetricPolicy.FAIL, ComparisonKind.INCOMPATIBLE),
        (MissingMetricPolicy.ERROR, ComparisonKind.ERROR),
        (MissingMetricPolicy.SKIP, ComparisonKind.UNCHANGED),
    ],
)
def test_missing_metric_policy_is_explicit(
    policy: MissingMetricPolicy, expected: ComparisonKind
) -> None:
    baseline = run("base", [trial("a")])
    candidate = run("cand", [trial("a")])
    summary = compare_runs(
        baseline,
        candidate,
        rules=(MetricRule("tokens.total", missing=policy),),
    )
    comparison = summary.case("a")
    assert comparison is not None
    assert comparison.kind is expected
    assert summary.passed is (expected is ComparisonKind.UNCHANGED)


def test_identity_mismatch_prevents_silent_comparison() -> None:
    baseline = run("base", [trial("a")], metadata={"dataset.id": "v1", "agent.id": "a"})
    candidate = run("cand", [trial("a")], metadata={"dataset.id": "v2", "agent.id": "a"})
    summary = compare_runs(baseline, candidate)
    assert len(summary.identity_mismatches) == 1
    mismatch = summary.identity_mismatches[0]
    assert mismatch.key == "dataset.id"
    assert mismatch.baseline == "v1"
    assert mismatch.candidate == "v2"
    assert not summary.passed


def test_added_and_removed_cases_are_explicit() -> None:
    baseline = run("base", [trial("old")])
    candidate = run("cand", [trial("new")])
    summary = compare_runs(baseline, candidate)
    assert compared(summary, "old").kind is ComparisonKind.REMOVED
    assert compared(summary, "new").kind is ComparisonKind.ADDED
    assert not summary.passed

    relaxed = compare_runs(baseline, candidate, require_same_cases=False)
    assert relaxed.passed


def test_execution_and_assessment_failures_are_distinct() -> None:
    baseline = run("base", [trial("a", assessment_error=True)])
    candidate = run(
        "cand",
        [trial("a", passed=False, status=EvalExecutionStatus.AGENT_ERROR)],
    )
    summary = compare_runs(baseline, candidate)
    comparison = summary.case("a")
    assert comparison is not None
    assert comparison.baseline_assessment_errors == 1
    assert comparison.candidate_execution_failures == 1
    assert comparison.candidate_assessment_errors == 0
    assert "candidate had more execution failures" in comparison.reasons


def test_aggregate_and_comparison_artifacts_round_trip() -> None:
    baseline = run("base", [trial("a", metrics=EvalMetrics({"x": metric("x", 1)}))])
    candidate = run("cand", [trial("a", metrics=EvalMetrics({"x": metric("x", 2)}))])
    aggregate = aggregate_run(baseline)
    restored_aggregate = RunAggregate.from_dict(aggregate.to_dict())
    assert restored_aggregate.to_dict() == aggregate.to_dict()

    summary = compare_runs(baseline, candidate, rules=(MetricRule("x"),))
    restored = ComparisonSummary.from_dict(summary.to_dict())
    assert restored.to_dict() == summary.to_dict()
    assert restored.passed is summary.passed


def context(output: EvalOutput | None = None) -> EvalContext:
    case = EvalCase.qa("case", "question")
    return EvalContext(
        case=case,
        trial_index=0,
        turn=case.turns[0],
        turn_index=0,
        output=output,
        metrics=EvalMetrics(),
    )


def test_state_predicate_checks_structured_output() -> None:
    predicate = StatePredicate("result.count", 2)
    passed = predicate.evaluate(context(EvalOutput(value={"result": {"count": 2}})))
    assert passed.status is EvalStatus.PASS

    failed = predicate.evaluate(context(EvalOutput(value={"result": {"count": 3}})))
    assert failed.status is EvalStatus.FAIL
    assert failed.evidence["observed"] == 3

    absent = predicate.evaluate(context(EvalOutput(value={"result": {}})))
    assert absent.status is EvalStatus.FAIL
    assert absent.evidence["found"] is False

    typed = StatePredicate("flag", 1).evaluate(context(EvalOutput(value={"flag": True})))
    assert typed.status is EvalStatus.FAIL


def test_optional_judge_adapter_is_provider_free() -> None:
    class FakeJudge:
        name = "fake-judge"

        def judge(self, current: EvalContext) -> JudgeVerdict:
            assert current.output is not None
            return JudgeVerdict(
                passed=current.output.text == "good",
                reasoning="checked the answer",
                score=1.0,
                evidence={"criterion": "answer"},
            )

    evaluator = JudgeEvaluator(FakeJudge())
    passed = evaluator.evaluate(context(EvalOutput(text="good")))
    assert passed.status is EvalStatus.PASS
    assert passed.evaluator == "fake-judge"
    assert passed.score == 1.0
    assert passed.evidence["judge"] == "fake-judge"
    assert passed.evidence["criterion"] == "answer"

    failed = evaluator.evaluate(context(EvalOutput(text="bad")))
    assert failed.status is EvalStatus.FAIL
    assert failed.message == "checked the answer"

    class BrokenJudge:
        name = "broken"

        def judge(self, current: EvalContext) -> JudgeVerdict:
            raise RuntimeError("provider payload text")

    broken = JudgeEvaluator(BrokenJudge()).evaluate(context(EvalOutput(text="x")))
    assert broken.status is EvalStatus.ERROR
    assert "provider payload text" not in str(broken.to_dict())


def test_case_aggregate_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        CaseAggregate(
            case_id="a",
            trial_count=1,
            passed_trials=2,
            execution_failures=0,
            assessment_errors=0,
            passed=True,
        )
