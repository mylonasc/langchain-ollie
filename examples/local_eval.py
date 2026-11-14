"""Run one local, trace-derived evaluation: python examples/local_eval.py."""

from local_agent import build_agent

from langgraph_instrumentation import (
    EvalCase,
    EvalRunner,
    EvalTurn,
    ExactText,
    LangGraphEvalAdapter,
    MaxMetric,
    ToolCalled,
)


def main() -> None:
    case = EvalCase(
        case_id="add-two-numbers",
        name="Add two numbers",
        turns=(
            EvalTurn.user(
                "Add 2 and 3",
                checks=(ExactText("Result: 5"), ToolCalled("add")),
            ),
        ),
        checks=(MaxMetric("llm.call_count", 2),),
    )
    run = EvalRunner(LangGraphEvalAdapter(build_agent())).run([case])
    trial = run.trials[0]
    turn = trial.turns[0]
    print(f"Case {trial.case_id}: {'PASS' if trial.passed else 'FAIL'}")
    for name in ("elapsed_ns", "tokens.total", "llm.call_count", "tool.call_count"):
        metric = turn.metrics.get(name)
        if metric is not None:
            print(f"  {name}: {metric.value} {metric.unit} ({metric.completeness.value})")
    print(f"  trace: {', '.join(turn.trace_ids) or 'none'}")
    if not run.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
