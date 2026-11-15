"""Offline evaluation contracts, execution, metrics and dataset tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from langchain_ollie import (
    CapturePolicy,
    EvalCase,
    EvalOutput,
    EvalRun,
    EvalRunner,
    EvalStatus,
    EvalTurn,
    ExactText,
    FunctionEvaluator,
    LangGraphEvalAdapter,
    MaxMetric,
    MetricCompleteness,
    RecordingLimits,
    RecordingSession,
    SpanKind,
    ToolCalled,
    load_cases,
    save_cases,
)
from langchain_ollie.resources import ProcessReading, ProcessSampler


class EchoApp:
    """Minimal async/sync app that records the payload it received."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        content = payload["messages"][-1][1]
        return {"messages": [AIMessage(content=f"echo:{content}")]}

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return self.invoke(payload, config)


class SpanApp:
    """Create one explicit graph span per invocation."""

    def invoke(self, invocation: Any) -> EvalOutput:
        correlation = uuid4().hex
        invocation.session.recorder.start_span(
            correlation,
            "Graph: test",
            SpanKind.GRAPH,
            attributes={"ls_integration": "langgraph"},
        )
        invocation.session.recorder.end_span(correlation)
        return EvalOutput(value={"messages": []}, text="ok")


class PublicSurfaceApp:
    """Realistic enough for one turn; spans and no token usage."""

    def invoke(self, invocation: Any) -> EvalOutput:
        correlation = uuid4().hex
        recorder = invocation.session.recorder
        recorder.start_span(correlation, "LLM: scripted", SpanKind.LLM)
        recorder.end_span(correlation)
        return EvalOutput(text="ok")


def metric(metrics: Any, name: str) -> Any:
    value = metrics.get(name)
    assert value is not None
    return value


def output_text(result: Any) -> str:
    assert result.output is not None
    assert result.output.text is not None
    return result.output.text


def test_single_turn_real_langgraph_run(local_agent) -> None:
    case = EvalCase(
        case_id="add",
        turns=(
            EvalTurn.user(
                "Add 2 and 3",
                checks=(ExactText("Result: 5"), ToolCalled("add")),
            ),
        ),
        checks=(MaxMetric("llm.call_count", 2),),
    )
    run = EvalRunner(LangGraphEvalAdapter(local_agent)).run([case])

    assert run.passed
    trial = run.trials[0]
    assert trial.execution_status.value == "completed"
    assert trial.passed
    turn = trial.turns[0]
    assert turn.output is not None
    assert turn.output.text == "Result: 5"
    assert metric(turn.metrics, "tokens.input").value == 5
    assert metric(turn.metrics, "tokens.output").value == 3
    assert metric(turn.metrics, "tokens.total").value == 8
    assert metric(turn.metrics, "llm.call_count").value == 2
    assert metric(turn.metrics, "tool.call_count").value == 1
    assert metric(turn.metrics, "tokens.total").source == "trace.metrics"
    assert metric(turn.metrics, "tokens.total").completeness is MetricCompleteness.COMPLETE
    assert len(turn.trace_ids) == 1
    assert trial.trace_ids == turn.trace_ids


def test_multi_turn_history_and_case_isolation() -> None:
    app = EchoApp()
    adapter = LangGraphEvalAdapter(app)

    def conversation_consistent(context: Any) -> bool:
        return (
            len(context.prior_turns) == 2
            and context.prior_turns[0].output.text == "echo:first"
            and context.prior_turns[1].output.text == "echo:second"
        )

    case_a = EvalCase(
        case_id="a",
        turns=(EvalTurn.user("first"), EvalTurn.user("second")),
        checks=(FunctionEvaluator("conversation", conversation_consistent),),
    )
    case_b = EvalCase(
        case_id="b",
        turns=(EvalTurn.user("first"), EvalTurn.user("second")),
    )
    run = EvalRunner(adapter).run([case_a, case_b])

    assert run.passed
    assert [len(payload["messages"]) for payload in app.payloads] == [1, 2, 1, 2]
    assert output_text(run.trials[0].turns[1]) == "echo:second"
    assert output_text(run.trials[1].turns[1]) == "echo:second"


def test_cases_use_fresh_sessions_and_stable_trace_references() -> None:
    run = EvalRunner(SpanApp()).run(
        [
            EvalCase(case_id="one", turns=(EvalTurn.user("x"),)),
            EvalCase(case_id="two", turns=(EvalTurn.user("y"),)),
        ]
    )
    first, second = (trial.trace_ids[0] for trial in run.trials)
    assert first != second
    assert metric(run.trials[0].turns[0].metrics, "recording.loss_count").value == 0


def test_execution_failure_is_explicit_and_skips_remaining_turns() -> None:
    class FailingApp:
        def invoke(self, invocation: Any) -> EvalOutput:
            raise RuntimeError("secret prompt text")

    run = EvalRunner(FailingApp()).run(
        [
            EvalCase(
                case_id="failure",
                turns=(EvalTurn.user("one"), EvalTurn.user("two")),
            )
        ]
    )
    trial = run.trials[0]
    assert trial.execution_status.value == "agent_error"
    assert trial.error is not None
    assert trial.error.exception_type.endswith("RuntimeError")
    assert "secret prompt text" not in str(trial.to_dict())
    assert trial.turns[0].execution_status.value == "agent_error"
    assert trial.turns[1].execution_status.value == "skipped"
    assert trial.turns[0].assessments == ()
    assert not trial.passed


def test_missing_token_evidence_is_not_reported_as_zero() -> None:
    run = EvalRunner(PublicSurfaceApp()).run(
        [
            EvalCase(
                case_id="missing",
                turns=(EvalTurn.user("x", checks=(MaxMetric("tokens.total", 1),)),),
            )
        ]
    )
    turn = run.trials[0].turns[0]
    metric = turn.metrics.get("tokens.total")
    assert metric is not None
    assert metric.value is None
    assert metric.completeness is MetricCompleteness.MISSING
    assert turn.assessments[0].status is EvalStatus.FAIL
    assert turn.assessments[0].message is not None
    assert "missing" in turn.assessments[0].message


def test_dataset_round_trip_json_jsonl_and_csv(tmp_path) -> None:
    case = EvalCase(
        case_id="multi",
        name="Multi-turn",
        description="Dataset round trip",
        tags=frozenset({"smoke", "local"}),
        metadata={"owner": "tests", "priority": 1},
        checks=(ToolCalled("add"),),
        turns=(
            EvalTurn(
                input={"messages": [["user", "Use add"]]},
                checks=(ExactText("5"),),
                metadata={"turn": 1},
                tags=frozenset({"first"}),
                timeout_seconds=2.5,
            ),
            EvalTurn.user(
                "thanks",
                checks=(MaxMetric("tokens.total", 100),),
            ),
        ),
    )
    for suffix in (".json", ".jsonl", ".csv"):
        destination = tmp_path / f"cases{suffix}"
        save_cases([case], destination)
        loaded = load_cases(destination)
        assert len(loaded) == 1
        restored = loaded[0]
        assert restored.case_id == case.case_id
        assert restored.name == case.name
        assert restored.description == case.description
        assert restored.tags == case.tags
        assert restored.metadata == case.metadata
        assert restored.to_dict() == case.to_dict()
        assert [turn.timeout_seconds for turn in restored.turns] == [2.5, None]
        assert [type(check).__name__ for check in restored.turns[0].checks] == ["ExactText"]
        assert [type(check).__name__ for check in restored.turns[1].checks] == ["MaxMetric"]
        assert [type(check).__name__ for check in restored.checks] == ["ToolCalled"]


def test_run_artifact_round_trip() -> None:
    run = EvalRunner(LangGraphEvalAdapter(EchoApp())).run(
        [
            EvalCase(
                case_id="artifact",
                turns=(EvalTurn.user("x", checks=(ExactText("echo:x"),)),),
            )
        ],
        metadata={"model": "scripted"},
    )
    payload = run.to_dict()
    restored = EvalRun.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.passed


@pytest.mark.asyncio
async def test_async_execution_supported() -> None:
    app = EchoApp()
    run = await EvalRunner(LangGraphEvalAdapter(app)).arun(
        [EvalCase(case_id="async", turns=(EvalTurn.user("hello"),))],
    )
    assert run.passed
    assert output_text(run.trials[0].turns[0]) == "echo:hello"


def test_custom_evaluator_exception_is_an_assessment_error() -> None:
    def explode(context: Any) -> bool:
        raise ValueError("private application detail")

    run = EvalRunner(LangGraphEvalAdapter(EchoApp())).run(
        [
            EvalCase(
                case_id="exploding-check",
                turns=(EvalTurn.user("x", checks=(FunctionEvaluator("explode", explode),)),),
            )
        ]
    )
    assessment = run.trials[0].turns[0].assessments[0]
    assert assessment.status is EvalStatus.ERROR
    assert "private application detail" not in str(assessment.to_dict())


def test_jsonl_dataset_rejects_unsupported_version(tmp_path) -> None:
    destination = tmp_path / "bad.jsonl"
    destination.write_text(
        '{"kind": "langgraph_instrumentation.eval_dataset_header", "schema_version": "99"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported evaluation dataset version"):
        load_cases(destination)


def test_runner_uses_bounded_capture_by_default(local_agent) -> None:
    run = EvalRunner(
        LangGraphEvalAdapter(local_agent),
        capture=CapturePolicy.metadata(),
    ).run([EvalCase(case_id="metadata", turns=(EvalTurn.user("Add 2 and 3"),))])
    assert run.metadata["eval.capture_mode"] == "metadata"
    assert metric(run.trials[0].turns[0].metrics, "tokens.total").value == 8


class NodeInterrupt(Exception):
    """LangGraph-style interruption; the name drives the execution status."""


def test_timeout_and_interrupt_map_to_explicit_statuses() -> None:
    class TimeoutApp:
        def invoke(self, invocation: Any) -> EvalOutput:
            raise TimeoutError("deadline exceeded")

    class InterruptApp:
        def invoke(self, invocation: Any) -> EvalOutput:
            raise NodeInterrupt("human review requested")

    timeout_run = EvalRunner(TimeoutApp()).run(
        [EvalCase(case_id="timeout", turns=(EvalTurn.user("one"), EvalTurn.user("two")))]
    )
    timeout_trial = timeout_run.trials[0]
    assert timeout_trial.execution_status.value == "timeout"
    assert timeout_trial.turns[0].execution_status.value == "timeout"
    assert timeout_trial.turns[1].execution_status.value == "skipped"
    assert not timeout_run.passed

    interrupt_run = EvalRunner(InterruptApp()).run(
        [EvalCase(case_id="interrupt", turns=(EvalTurn.user("one"),))]
    )
    interrupt_trial = interrupt_run.trials[0]
    assert interrupt_trial.execution_status.value == "interrupted"
    assert interrupt_trial.error is not None
    assert "NodeInterrupt" in interrupt_trial.error.exception_type
    assert not interrupt_run.passed


class UnclosableSession(RecordingSession):
    """Simulate a session whose shutdown cannot complete."""

    def close(self, timeout: float | None = None) -> bool:
        return False


def test_unclosable_session_marks_trial_incomplete() -> None:
    run = EvalRunner(SpanApp(), session_factory=UnclosableSession).run(
        [EvalCase(case_id="incomplete", turns=(EvalTurn.user("x"),))]
    )
    trial = run.trials[0]
    assert trial.turns[0].execution_status.value == "completed"
    assert trial.execution_status.value == "incomplete"
    assert trial.error is not None
    assert trial.error.exception_type == "RecordingSession.close"
    assert not trial.passed


def test_supplied_environment_is_distinguishable_from_extracted() -> None:
    run = EvalRunner(LangGraphEvalAdapter(EchoApp())).run(
        [EvalCase(case_id="env", turns=(EvalTurn.user("x"),))],
        environment={"git_commit": "abc123", "model.id": "scripted"},
    )
    assert run.environment["supplied"] == {"git_commit": "abc123", "model.id": "scripted"}
    assert "python" in run.environment
    packages = run.environment["packages"]
    assert isinstance(packages, Mapping)
    assert packages["langchain-ollie"] == "0.1.0"


class TwoTraceApp:
    """Record two independent root spans, hence two traces, in one turn."""

    def invoke(self, invocation: Any) -> EvalOutput:
        recorder = invocation.session.recorder
        for name in ("Graph: first", "Graph: second"):
            correlation = uuid4().hex
            recorder.start_span(
                correlation, name, SpanKind.GRAPH, attributes={"ls_integration": "langgraph"}
            )
            recorder.end_span(correlation)
        return EvalOutput(text="ok")


def test_multiple_traces_in_one_turn_aggregate_deterministically() -> None:
    run = EvalRunner(TwoTraceApp()).run(
        [EvalCase(case_id="two-traces", turns=(EvalTurn.user("x"),))]
    )
    turn = run.trials[0].turns[0]
    assert len(turn.trace_ids) == 2
    assert len(set(turn.trace_ids)) == 2
    assert metric(turn.metrics, "span.count").value == 2
    assert run.trials[0].trace_ids == turn.trace_ids


class ScriptedSource:
    """Deterministic process readings without touching psutil."""

    def __init__(self) -> None:
        self.readings = [
            ProcessReading(cpu_seconds=10.0, rss_bytes=100),
            ProcessReading(cpu_seconds=12.0, rss_bytes=200),
        ]

    def read(self) -> ProcessReading:
        return self.readings.pop(0)


class SampledApp:
    """Attach an optional process sampler with a deterministic fake source."""

    def invoke(self, invocation: Any) -> EvalOutput:
        ticks = iter([1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000])
        sampler = ProcessSampler(
            invocation.session.recorder,
            source=ScriptedSource(),
            monotonic_ns=lambda: next(ticks),
        ).start(background=False)
        assert sampler.sample_once()
        assert sampler.sample_once()
        assert sampler.close()
        return EvalOutput(text="ok")


def test_process_sampler_metrics_are_process_scoped_and_optional() -> None:
    run = EvalRunner(SampledApp()).run([EvalCase(case_id="sampled", turns=(EvalTurn.user("x"),))])
    turn = run.trials[0].turns[0]
    samples = metric(turn.metrics, "process.samples")
    assert samples.value == 2
    assert samples.scope.value == "process"
    assert samples.source == "process sampler"
    rss = metric(turn.metrics, "process.rss_bytes")
    assert rss.value == 200
    assert rss.scope.value == "process"
    assert "not per-agent" in (rss.detail or "")
    cpu = metric(turn.metrics, "process.cpu_seconds")
    assert cpu.value == 2.0
    assert cpu.scope.value == "process"


def test_recording_loss_is_visible_in_metrics(local_agent) -> None:
    run = EvalRunner(
        LangGraphEvalAdapter(local_agent),
        limits=RecordingLimits(max_spans=1),
    ).run([EvalCase(case_id="lossy", turns=(EvalTurn.user("Add 2 and 3"),))])
    turn = run.trials[0].turns[0]
    loss = metric(turn.metrics, "recording.loss_count")
    assert loss.value is not None and loss.value > 0
    assert "loss reported by" in (loss.detail or "")


def test_custom_evaluator_can_read_traces_and_metrics() -> None:
    def inspect_evidence(context: Any) -> bool:
        kinds = {span.kind for bundle in context.traces for span in bundle.spans}
        count = context.metrics.get("span.count")
        return (
            SpanKind.GRAPH in kinds
            and count is not None
            and count.value == 1
            and context.output is not None
            and context.output.text == "ok"
        )

    run = EvalRunner(SpanApp()).run(
        [
            EvalCase(
                case_id="inspect",
                turns=(
                    EvalTurn.user("x", checks=(FunctionEvaluator("inspect", inspect_evidence),)),
                ),
            )
        ]
    )
    assert run.passed
    assert run.trials[0].turns[0].assessments[0].status is EvalStatus.PASS
