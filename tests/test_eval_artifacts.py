"""Offline persistence tests: canonical artifacts, versions and JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langgraph_instrumentation import (
    ComparisonSummary,
    EvalAssessment,
    EvalExecutionStatus,
    EvalMetric,
    EvalMetrics,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurnResult,
    MetricCompleteness,
    MetricRule,
    MetricScope,
    RunAggregate,
    aggregate_run,
    compare_runs,
    load_aggregate,
    load_any,
    load_artifacts,
    load_comparison,
    load_run,
    save_aggregate,
    save_artifacts,
    save_comparison,
    save_run,
)
from langgraph_instrumentation.evals.comparison import ComparisonKind


def _run(run_id: str, passed: bool, tokens: int) -> EvalRun:
    status = EvalStatus.PASS if passed else EvalStatus.FAIL
    turn = EvalTurnResult(
        index=0,
        input={"messages": [["user", "x"]]},
        execution_status=EvalExecutionStatus.COMPLETED,
        assessments=(EvalAssessment("check", status),),
        metrics=EvalMetrics(
            {
                "tokens.total": EvalMetric(
                    name="tokens.total",
                    value=tokens,
                    unit="tokens",
                    source="tests",
                    scope=MetricScope.CASE,
                    completeness=MetricCompleteness.COMPLETE,
                )
            }
        ),
    )
    trial = EvalTrial(
        case_id="a",
        trial_index=0,
        execution_status=EvalExecutionStatus.COMPLETED,
        turns=(turn,),
        metrics=turn.metrics,
    )
    return EvalRun(
        run_id=run_id,
        trials=(trial,),
        started_unix_ns=0,
        finished_unix_ns=1,
    )


def test_run_artifact_is_byte_stable(tmp_path: Path) -> None:
    destination = tmp_path / "run.json"
    save_run(_run("base", True, 100), destination)
    first = destination.read_bytes()
    restored = load_run(destination)
    save_run(restored, destination)
    assert destination.read_bytes() == first
    assert restored.to_dict() == json.loads(first.decode())


def test_aggregate_and_comparison_artifacts_round_trip(tmp_path: Path) -> None:
    baseline, candidate = _run("base", True, 100), _run("cand", True, 120)
    aggregate = aggregate_run(baseline)
    summary = compare_runs(baseline, candidate, rules=(MetricRule("tokens.total"),))

    aggregate_path, comparison_path = tmp_path / "agg.json", tmp_path / "cmp.json"
    save_aggregate(aggregate, aggregate_path)
    save_comparison(summary, comparison_path)
    assert isinstance(load_aggregate(aggregate_path), RunAggregate)
    restored = load_comparison(comparison_path)
    assert isinstance(restored, ComparisonSummary)
    assert restored.to_dict() == summary.to_dict()
    compared = restored.case("a")
    assert compared is not None
    assert compared.kind is ComparisonKind.REGRESSED


def test_jsonl_preserves_mixed_artifacts(tmp_path: Path) -> None:
    baseline = _run("base", True, 100)
    aggregate = aggregate_run(baseline)
    summary = compare_runs(baseline, _run("cand", True, 100))
    destination = tmp_path / "all.jsonl"
    save_artifacts([baseline, aggregate, summary], destination)
    restored = load_artifacts(destination)
    assert [type(item) for item in restored] == [EvalRun, RunAggregate, ComparisonSummary]
    assert restored[0].to_dict() == baseline.to_dict()  # type: ignore[union-attr]


def test_load_any_dispatches_on_kind(tmp_path: Path) -> None:
    run_path, aggregate_path = tmp_path / "run.json", tmp_path / "agg.json"
    run = _run("base", True, 100)
    save_run(run, run_path)
    save_aggregate(aggregate_run(run), aggregate_path)
    assert isinstance(load_any(run_path), EvalRun)
    assert isinstance(load_any(aggregate_path), RunAggregate)


def test_unsupported_versions_fail_explicitly(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    save_run(_run("base", True, 1), run_path)
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "99"
    run_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="unsupported evaluation run schema version|invalid evaluation run"
    ):
        load_run(run_path)

    comparison_path = tmp_path / "cmp.json"
    comparison_path.write_text(
        json.dumps({"kind": "langgraph_instrumentation.eval_comparison", "schema_version": "99"}),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="unsupported comparison schema version|invalid baseline comparison"
    ):
        load_comparison(comparison_path)


def test_wrong_kind_and_suffix_fail_explicitly(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    save_run(_run("base", True, 1), run_path)
    with pytest.raises(ValueError, match="not a .*aggregate"):
        load_aggregate(run_path)
    with pytest.raises(ValueError, match="must end with .json"):
        load_run(tmp_path / "run.jsonl")
    with pytest.raises(ValueError, match="must end with .jsonl"):
        save_artifacts([_run("base", True, 1)], tmp_path / "all.json")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_artifacts(empty)


def test_artifacts_reference_traces_without_embedding_payloads(tmp_path: Path) -> None:
    import dataclasses

    run = _run("base", True, 5)
    turn = run.trials[0].turns[0]
    referenced_turn = dataclasses.replace(turn, trace_ids=("0123456789abcdef0123456789abcdef",))
    trial = dataclasses.replace(run.trials[0], turns=(referenced_turn,))
    run = dataclasses.replace(run, trials=(trial,))
    destination = tmp_path / "run.json"
    save_run(run, destination)
    text = destination.read_text(encoding="utf-8")
    assert "0123456789abcdef0123456789abcdef" in text
    payload = json.loads(text)
    turn_payload = payload["trials"][0]["turns"][0]
    assert turn_payload["trace_ids"] == ["0123456789abcdef0123456789abcdef"]
    assert "spans" not in turn_payload
    assert "events" not in text
