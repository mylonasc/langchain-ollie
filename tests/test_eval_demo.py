"""End-to-end demonstration gate: quality-preserving optimization (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from langchain_ollie import (
    EvalRunner,
    LangGraphEvalAdapter,
    MetricRule,
    compare_runs,
    load_cases,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch) -> Any:
    import runpy

    monkeypatch.syspath_prepend(str(EXAMPLES))
    namespace = runpy.run_path(str(EXAMPLES / "eval_demo.py"))
    return namespace


def _builders(demo: Any) -> Any:
    return demo["build_baseline_app"], demo["build_optimized_app"]


def test_demo_gate_fails_baseline_and_passes_optimized(demo: Any) -> None:
    build_baseline, build_optimized = _builders(demo)
    golden = load_cases(EXAMPLES / "eval_demo_cases_v2.json")
    baseline_run = EvalRunner(LangGraphEvalAdapter(build_baseline())).run(golden)
    assert not baseline_run.passed
    budget_failures = [
        assessment
        for trial in baseline_run.trials
        for turn in trial.turns
        for assessment in (*turn.assessments, *trial.assessments)
        if assessment.evaluator in ("max_metric",) and not assessment.passed
    ]
    assert budget_failures, "baseline must fail the efficiency budgets"

    optimized_run = EvalRunner(LangGraphEvalAdapter(build_optimized())).run(golden)
    assert optimized_run.passed


def test_demo_comparison_shows_quality_preserving_improvement(demo: Any) -> None:
    build_baseline, build_optimized = _builders(demo)
    golden = load_cases(EXAMPLES / "eval_demo_cases_v2.json")
    baseline_run = EvalRunner(LangGraphEvalAdapter(build_baseline())).run(golden)
    optimized_run = EvalRunner(LangGraphEvalAdapter(build_optimized())).run(golden)

    comparison = compare_runs(baseline_run, optimized_run, rules=(MetricRule("tokens.total"),))
    assert comparison.passed
    assert "efficient-fetch" in comparison.improved_case_ids

    baseline_tokens = baseline_run.trials[0].metrics.get("tokens.total")
    optimized_tokens = optimized_run.trials[0].metrics.get("tokens.total")
    assert baseline_tokens is not None and baseline_tokens.value == 52
    assert optimized_tokens is not None and optimized_tokens.value == 28


def test_demo_gate_rejects_regression(demo: Any) -> None:
    build_baseline, build_optimized = _builders(demo)
    golden = load_cases(EXAMPLES / "eval_demo_cases_v2.json")
    baseline_run = EvalRunner(LangGraphEvalAdapter(build_baseline())).run(golden)
    optimized_run = EvalRunner(LangGraphEvalAdapter(build_optimized())).run(golden)
    reversed_comparison = compare_runs(
        optimized_run, baseline_run, rules=(MetricRule("tokens.total"),)
    )
    assert not reversed_comparison.passed
    assert "efficient-fetch" in reversed_comparison.regressed_case_ids


def test_demo_report_shows_tradeoff_and_missing_evidence(demo: Any, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from langchain_ollie import render_report

    build_baseline, build_optimized = _builders(demo)
    golden = load_cases(EXAMPLES / "eval_demo_cases_v2.json")
    baseline_run = EvalRunner(LangGraphEvalAdapter(build_baseline())).run(golden)
    optimized_run = EvalRunner(LangGraphEvalAdapter(build_optimized())).run(golden)
    comparison = compare_runs(baseline_run, optimized_run, rules=(MetricRule("tokens.total"),))
    document = render_report(optimized_run, comparison=comparison)
    assert "efficient-fetch" in document
    assert "tokens.total" in document
    assert "missing" in document
    report = tmp_path / "report.html"
    report.write_text(document, encoding="utf-8")
    assert report.stat().st_size > 0
