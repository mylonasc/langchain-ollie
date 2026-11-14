"""Offline CLI tests: thin Python-API surface, exit codes, offline HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from langgraph_instrumentation import (
    EvalAssessment,
    EvalCase,
    EvalExecutionStatus,
    EvalMetrics,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurn,
    EvalTurnResult,
    ExactText,
    save_cases,
    save_run,
)
from langgraph_instrumentation.evals.cli import main as evals_main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _dataset(path: Path) -> Path:
    destination = path / "cases.json"
    save_cases(
        [
            EvalCase(
                case_id="add",
                turns=(EvalTurn.user("Add 2 and 3", checks=(ExactText("Result: 5"),)),),
            )
        ],
        destination,
    )
    return destination


def _scripted_module(path: Path) -> None:
    (path / "tiny_agent.py").write_text(
        "from langchain_core.messages import AIMessage\n"
        "\n"
        "class TinyApp:\n"
        "    def invoke(self, payload, config=None):\n"
        '        return {"messages": [AIMessage(content="Result: 5")]}\n'
        "\n"
        "def build():\n"
        "    return TinyApp()\n",
        encoding="utf-8",
    )


def _run(run_id: str, status: EvalStatus, extra_ns: int = 1) -> EvalRun:
    turn = EvalTurnResult(
        index=0,
        input={"messages": [["user", "x"]]},
        execution_status=EvalExecutionStatus.COMPLETED,
        assessments=(EvalAssessment("check", status),),
        metrics=EvalMetrics(),
    )
    trial = EvalTrial(
        case_id="a",
        trial_index=0,
        execution_status=EvalExecutionStatus.COMPLETED,
        turns=(turn,),
    )
    return EvalRun(run_id=run_id, trials=(trial,), started_unix_ns=0, finished_unix_ns=extra_ns)


def test_run_executes_real_agent_and_saves_canonical_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    dataset = _dataset(tmp_path)
    out = tmp_path / "run.json"
    code = evals_main(
        ["run", str(dataset), "--agent", "local_agent:build_agent", "--out", str(out)]
    )
    assert code == 0
    assert out.is_file()
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "run": str(out),
        "run_id": document["run_id"],
        "passed": True,
        "trials": 1,
        "failed_cases": [],
    }


def test_run_supports_module_factory_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    _scripted_module(tmp_path)
    out = tmp_path / "run.json"
    code = evals_main(
        ["run", str(_dataset(tmp_path)), "--agent", "tiny_agent:build", "--out", str(out)]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_run_rejects_unknown_agent(tmp_path: Path, capsys: Any) -> None:
    code = evals_main(
        [
            "run",
            str(_dataset(tmp_path)),
            "--agent",
            "no_such_module:build",
            "--out",
            str(tmp_path / "run.json"),
        ]
    )
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_cases_lists_and_inspects(tmp_path: Path, capsys: Any) -> None:
    dataset = _dataset(tmp_path)
    assert evals_main(["cases", str(dataset)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["cases"] == [{"case_id": "add", "turns": 1, "checks": 1}]

    assert evals_main(["cases", str(dataset), "--case", "add"]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["case_id"] == "add"

    assert evals_main(["cases", str(dataset), "--case", "missing"]) == 2


def test_show_reports_saved_run(tmp_path: Path, capsys: Any) -> None:
    run_path = tmp_path / "run.json"
    save_run(_run("base", EvalStatus.PASS), run_path)
    assert evals_main(["show", str(run_path)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["passed"] is True
    assert document["cases"][0] == {
        "case_id": "a",
        "passed": True,
        "pass_rate": 1.0,
        "trials": 1,
        "execution_failures": 0,
        "assessment_errors": 0,
    }
    assert evals_main(["show", str(tmp_path / "absent.json")]) == 2


def test_compare_gate_exit_codes_and_stable_output(tmp_path: Path, capsys: Any) -> None:
    base, same, worse = (
        tmp_path / "base.json",
        tmp_path / "same.json",
        tmp_path / "worse.json",
    )
    save_run(_run("base", EvalStatus.PASS, 1), base)
    save_run(_run("same", EvalStatus.PASS, 1), same)
    save_run(_run("worse", EvalStatus.FAIL, 1), worse)

    assert evals_main(["compare", str(base), str(same)]) == 0
    first = capsys.readouterr().out
    assert evals_main(["compare", str(base), str(same)]) == 0
    assert capsys.readouterr().out == first

    assert evals_main(["compare", str(base), str(worse)]) == 1
    regressed = json.loads(capsys.readouterr().out)
    assert regressed["regressed_case_ids"] == ["a"]

    out = tmp_path / "comparison.json"
    assert evals_main(["compare", str(base), str(worse), "--out", str(out)]) == 1
    assert out.is_file()

    assert evals_main(["compare", str(base), str(worse), "--rule", "x:bogus"]) == 2
    assert evals_main(["compare", str(base), str(worse), "--rule", "x"]) == 2


def test_validate_accepts_and_rejects(tmp_path: Path, capsys: Any) -> None:
    dataset = _dataset(tmp_path)
    assert evals_main(["validate", str(dataset)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "file": str(dataset),
        "kind": "dataset",
        "valid": True,
        "cases": 1,
    }

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "nope", "schema_version": "1"}), encoding="utf-8")
    assert evals_main(["validate", str(bad)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "langgraph_instrumentation.eval_dataset",
                "cases": [
                    {
                        "case_id": "same",
                        "name": None,
                        "description": None,
                        "tags": [],
                        "metadata": {},
                        "turns": [
                            {
                                "input": None,
                                "messages": [["user", "x"]],
                                "checks": [],
                                "metadata": {},
                                "tags": [],
                                "timeout_seconds": None,
                            }
                        ],
                        "checks": [],
                    },
                    {
                        "case_id": "same",
                        "name": None,
                        "description": None,
                        "tags": [],
                        "metadata": {},
                        "turns": [
                            {
                                "input": None,
                                "messages": [["user", "y"]],
                                "checks": [],
                                "metadata": {},
                                "tags": [],
                                "timeout_seconds": None,
                            }
                        ],
                        "checks": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert evals_main(["validate", str(duplicate)]) == 1
    assert "duplicate case_id" in json.loads(capsys.readouterr().out)["error"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "langgraph_instrumentation.eval_dataset",
                "cases": [
                    {
                        "case_id": "broken",
                        "name": None,
                        "description": None,
                        "tags": [],
                        "metadata": {},
                        "turns": [
                            {
                                "input": None,
                                "messages": [],
                                "checks": [],
                                "metadata": {},
                                "tags": [],
                                "timeout_seconds": None,
                            }
                        ],
                        "checks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert evals_main(["validate", str(malformed)]) == 1

    assert evals_main(["validate", str(tmp_path / "missing.json")]) == 2


def test_report_writes_offline_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    run_path = tmp_path / "run.json"
    assert (
        evals_main(
            [
                "run",
                str(_dataset(tmp_path)),
                "--agent",
                "local_agent:build_agent",
                "--out",
                str(run_path),
            ]
        )
        == 0
    )
    report = tmp_path / "report.html"
    assert evals_main(["report", str(run_path), "--out", str(report)]) == 0
    document = report.read_text(encoding="utf-8")
    assert "<html" in document
    assert "add" in document
    assert "tokens.total" in document
    assert "0123456789abcdef" not in document
    assert "http" not in document
    assert "<script" not in document


def _candidate_cases(path: Path) -> Path:
    from langgraph_instrumentation import EvalCase, EvalTurn, ExactText, save_cases

    destination = path / "candidate.json"
    save_cases(
        [
            EvalCase(
                case_id="outage-window",
                turns=(EvalTurn.user("What caused the outage?", checks=(ExactText("ok"),)),),
            )
        ],
        destination,
    )
    return destination


def test_inspect_reports_failures_and_turn_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    run_path = tmp_path / "run.json"
    assert (
        evals_main(
            [
                "run",
                str(_dataset(tmp_path)),
                "--agent",
                "local_agent:build_agent",
                "--out",
                str(run_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert evals_main(["inspect", str(run_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["failed_trials"] == 0

    assert evals_main(["inspect", str(run_path), "--case", "add", "--turn", "0"]) == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["case_id"] == "add"
    assert evidence["missing_trace_ids"] != []
    assert evidence["calls"] == []

    assert evals_main(["inspect", str(run_path), "--turn", "0"]) == 2


def test_proposal_lifecycle_through_cli(tmp_path: Path, capsys: Any) -> None:
    dataset = _dataset(tmp_path)
    candidate = _candidate_cases(tmp_path)
    proposal = tmp_path / "proposal.json"

    code = evals_main(
        [
            "propose",
            "--base-dataset",
            str(dataset),
            "--candidate-cases",
            str(candidate),
            "--rationale",
            "Cover the outage window seen in the failure summary.",
            "--author",
            "agent",
            "--out",
            str(proposal),
        ]
    )
    assert code == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "candidate"
    assert created["validation_passed"] is True
    capsys.readouterr()

    assert evals_main(["proposal-validate", str(proposal), "--base-dataset", str(dataset)]) == 0
    capsys.readouterr()

    reviewed = tmp_path / "reviewed.json"
    assert (
        evals_main(
            [
                "proposal-review",
                str(proposal),
                "--decision",
                "accept",
                "--reviewer",
                "sme",
                "--note",
                "Reproduces the outage.",
                "--out",
                str(reviewed),
            ]
        )
        == 0
    )

    golden = tmp_path / "golden.json"
    assert evals_main(["apply-proposal", str(dataset), str(reviewed), "--out", str(golden)]) == 0
    capsys.readouterr()
    assert evals_main(["cases", str(golden)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [case["case_id"] for case in listed["cases"]] == ["add", "outage-window"]

    assert evals_main(["apply-proposal", str(dataset), str(proposal), "--out", str(golden)]) == 2


def test_inspect_resolves_calls_from_read_only_trace_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, local_agent: Any
) -> None:
    from langgraph_instrumentation import (
        EvalRunner,
        LangGraphEvalAdapter,
        SQLiteTraceStore,
    )

    monkeypatch.syspath_prepend(str(EXAMPLES))
    traces_db = tmp_path / "traces.db"
    run = EvalRunner(
        LangGraphEvalAdapter(local_agent),
        store_factory=lambda: SQLiteTraceStore(traces_db),
    ).run(
        [
            EvalCase(
                case_id="add",
                turns=(EvalTurn.user("Add 2 and 3"),),
            )
        ]
    )
    run_path = tmp_path / "run.json"
    save_run(run, run_path)
    capsys.readouterr()

    assert (
        evals_main(
            [
                "inspect",
                str(run_path),
                "--case",
                "add",
                "--turn",
                "0",
                "--traces",
                str(traces_db),
            ]
        )
        == 0
    )
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["missing_trace_ids"] == []
    assert {call["kind"] for call in evidence["calls"]} == {"llm", "tool"}
