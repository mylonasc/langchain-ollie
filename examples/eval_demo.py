"""Reproducible end-to-end evaluation demo: baseline vs optimized agent.

Run offline from the repository root:

    uv run --no-sync python examples/eval_demo.py
    uv run --no-sync python examples/eval_demo.py --out-dir /tmp/eval-demo

The story: a lookup-based Q&A agent answers correctly, but the baseline
implementation fetches every fact twice. A resource review of the recorded
traces spots the duplicated tool calls. An external coding agent (working only
through the public evaluation API) proposes a golden regression case that pins
single-fetch behavior; after explicit review and acceptance, an optimized agent
implementation passes the full golden suite with lower token consumption while
answer quality is unchanged.

Every step below uses published Python interfaces. The script exits non-zero
if any expectation is violated, so it doubles as a local regression gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from scripted_model import ScriptedModel

from langchain_ollie import (
    EvalCase,
    EvalRunner,
    EvalTurn,
    ExactText,
    LangGraphEvalAdapter,
    MaxMetric,
    MetricRule,
    RecordingSession,
    ToolCalled,
    apply_proposal,
    compare_runs,
    load_cases,
    propose,
    render_report,
    review_proposal,
    summarize_failures,
    turn_evidence,
)

EXAMPLES = Path(__file__).resolve().parent
V1_DATASET = EXAMPLES / "eval_demo_cases_v1.json"
V2_DATASET = EXAMPLES / "eval_demo_cases_v2.json"
TOKEN_BUDGET = 20


@tool
def lookup(query: str) -> str:
    """Look up a fact locally; deterministic and offline."""
    return f"fact:{query}=42"


def _responses(double_fetch: bool) -> tuple[AIMessage, ...]:
    first = (10, 2, 12) if double_fetch else (6, 1, 7)
    second = (12, 2, 14) if double_fetch else (6, 1, 7)

    def fetch(text: str) -> AIMessage:
        calls = {"name": "lookup", "args": {"query": "value"}}
        tool_calls = [
            {**calls, "id": "call-lookup-a"},
            {**calls, "id": "call-lookup-b"},
        ][: 2 if double_fetch else 1]
        return AIMessage(
            content="",
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": first[0],
                "output_tokens": first[1],
                "total_tokens": first[2],
            },
        )

    def answer(text: str) -> AIMessage:
        return AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": second[0],
                "output_tokens": second[1],
                "total_tokens": second[2],
            },
        )

    return (
        fetch(""),
        answer("Answer: 42"),
        fetch(""),
        answer("Confirmed: 42"),
        fetch(""),
        answer("Answer: 42"),
    )


def build_baseline_app() -> Any:
    """Answer correctly while fetching every fact twice."""
    return _build_app(double_fetch=True, name="DemoBaseline")


def build_optimized_app() -> Any:
    """Answer identically while fetching every fact once."""
    return _build_app(double_fetch=False, name="DemoOptimized")


def _build_app(*, double_fetch: bool, name: str) -> Any:
    model = ScriptedModel(responses=_responses(double_fetch)).bind_tools([lookup])

    def call_model(state: MessagesState, config: RunnableConfig):
        return {"messages": [model.invoke(state["messages"], config=config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([lookup]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile(name=name)


def v1_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            case_id="capital-lookup",
            name="Multi-turn lookup",
            turns=(
                EvalTurn.user("Look up the capital value.", checks=(ExactText("Answer: 42"),)),
                EvalTurn.user("Confirm it once more.", checks=(ExactText("Confirmed: 42"),)),
            ),
        ),
    )


def efficiency_case() -> EvalCase:
    """Golden regression case derived from the duplicated-fetch evidence."""
    return EvalCase(
        case_id="efficient-fetch",
        name="Single-fetch budget",
        turns=(
            EvalTurn.user(
                "What is the value?",
                checks=(
                    ExactText("Answer: 42"),
                    ToolCalled("lookup"),
                    MaxMetric("tool.call_count", 1),
                ),
            ),
        ),
        checks=(MaxMetric("tokens.total", TOKEN_BUDGET),),
    )


def diagnose(run: Any, bundles: Any) -> dict[str, Any]:
    """External-agent diagnosis using only public inspection interfaces."""
    summary = summarize_failures(run)
    calls: dict[str, int] = {}
    for trial in run.trials:
        for turn in trial.turns:
            evidence = turn_evidence(run, trial.case_id, turn.index, traces=bundles)
            for call in evidence.calls:
                calls[call.name] = calls.get(call.name, 0) + 1
    return {
        "failed_trials": summary.failed_trials,
        "tool_calls": calls,
        "trace_ids": [trace_id for trial in run.trials for trace_id in trial.trace_ids],
    }


def _require(condition: bool, message: str) -> None:
    print(("ok" if condition else "FAIL") + f" - {message}")
    if not condition:
        raise SystemExit(f"demo expectation violated: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencode/eval-demo"))
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = load_cases(V1_DATASET)
    _require(tuple(case.case_id for case in v1) == ("capital-lookup",), "v1 dataset loads")

    sessions: list[RecordingSession] = []

    def keep_session() -> RecordingSession:
        sessions.append(RecordingSession())
        return sessions[-1]

    baseline_run = EvalRunner(
        LangGraphEvalAdapter(build_baseline_app()), session_factory=keep_session
    ).run(v1)
    _require(baseline_run.passed, "baseline answers v1 correctly (quality passes)")

    bundles = {
        str(bundle.trace.trace_id): bundle for session in sessions for bundle in session.snapshot()
    }
    diagnosis = diagnose(baseline_run, bundles)
    print(f"diagnosis: {diagnosis}")
    _require(
        sum(diagnosis["tool_calls"].values()) >= 2,
        "trace evidence shows duplicated fetches",
    )

    proposal = propose(
        [efficiency_case()],
        author="external-agent",
        rationale=(
            "Recorded traces show two lookup calls per turn; pin single-fetch "
            "behavior with tool-count and token budgets."
        ),
        validate_against=v1,
    )
    _require(
        proposal.validation is not None and proposal.validation.passed,
        "candidate proposal validates",
    )
    reviewed = review_proposal(
        proposal, reviewer="sme", decision="accepted", note="Reproduces the waste."
    )
    golden = apply_proposal(v1, reviewed)
    _require(
        [case.case_id for case in golden] == [case.case_id for case in load_cases(V2_DATASET)],
        "accepted proposal reproduces the committed golden dataset",
    )

    candidate_run = EvalRunner(LangGraphEvalAdapter(build_baseline_app())).run(golden)
    _require(not candidate_run.passed, "baseline fails the new efficiency budgets")
    optimized_run = EvalRunner(LangGraphEvalAdapter(build_optimized_app())).run(golden)
    _require(optimized_run.passed, "optimized agent passes the golden suite")

    comparison = compare_runs(
        candidate_run,
        optimized_run,
        rules=(MetricRule("tokens.total"),),
    )
    _require(comparison.passed, "optimization improves without quality regression")
    improved = comparison.improved_case_ids
    _require("efficient-fetch" in improved, "efficiency case improved")
    baseline_tokens = candidate_run.trials[0].metrics.get("tokens.total")
    optimized_tokens = optimized_run.trials[0].metrics.get("tokens.total")
    assert baseline_tokens is not None and baseline_tokens.value is not None
    assert optimized_tokens is not None and optimized_tokens.value is not None
    print(
        f"tokens.total per trial: baseline={baseline_tokens.value} "
        f"optimized={optimized_tokens.value}"
    )

    report = out_dir / "report.html"
    report.write_text(
        render_report(optimized_run, comparison=comparison, title="Eval demo report"),
        encoding="utf-8",
    )
    print(f"report: {report}")
    print("demo gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
