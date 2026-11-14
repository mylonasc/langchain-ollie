"""Orchestrate reproducible case execution, recording and assessment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

from ..clock import Clock, ClockReading, SystemClock
from ..limits import RecordingLimits
from ..models import TraceBundle
from ..policies import CapturePolicy, RedactionPolicy
from ..session import RecordingSession
from .adapters import EvalInvocation, ExecutionAdapter
from .metrics import aggregate_turn_metrics, measure_traces
from .models import (
    EvalAssessment,
    EvalCase,
    EvalError,
    EvalExecutionStatus,
    EvalMetrics,
    EvalOutput,
    EvalRun,
    EvalStatus,
    EvalTrial,
    EvalTurn,
    EvalTurnResult,
)

_MAX_ERROR_MESSAGE = 500


class EvalRunner:
    """Execute cases locally with a fresh recording session per case trial."""

    def __init__(
        self,
        adapter: ExecutionAdapter,
        *,
        capture: CapturePolicy | None = None,
        redaction: RedactionPolicy | None = None,
        limits: RecordingLimits | None = None,
        trials: int = 1,
        session_factory: Any | None = None,
        store_factory: Any | None = None,
        clock: Clock | None = None,
        error_messages: bool = False,
    ) -> None:
        if not hasattr(adapter, "invoke") and not hasattr(adapter, "ainvoke"):
            raise TypeError("adapter must implement invoke() or ainvoke()")
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ValueError("trials must be a positive integer")
        self._adapter = adapter
        self._capture = capture or CapturePolicy.metadata()
        self._redaction = redaction
        self._limits = limits
        self._trials = trials
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._clock = clock or SystemClock()
        self._error_messages = error_messages

    def run(
        self,
        cases: Sequence[EvalCase],
        *,
        run_id: str | None = None,
        trials: int | None = None,
        metadata: Mapping[str, object] | None = None,
        environment: Mapping[str, object] | None = None,
    ) -> EvalRun:
        """Execute cases synchronously and return an immutable run."""
        case_list = _validate_cases(cases)
        trial_count = _validate_trials(trials, self._trials)
        identifier = run_id or uuid4().hex
        started = self._clock.now()
        results = [
            self._run_trial(identifier, case, index)
            for case in case_list
            for index in range(trial_count)
        ]
        finished = self._clock.now()
        return EvalRun(
            run_id=identifier,
            trials=tuple(results),
            started_unix_ns=started.unix_time_ns,
            finished_unix_ns=finished.unix_time_ns,
            metadata=_run_metadata(self._capture, trial_count, metadata),
            environment=_environment(environment),
        )

    async def arun(
        self,
        cases: Sequence[EvalCase],
        *,
        run_id: str | None = None,
        trials: int | None = None,
        metadata: Mapping[str, object] | None = None,
        environment: Mapping[str, object] | None = None,
    ) -> EvalRun:
        """Execute cases asynchronously when the adapter provides ``ainvoke``."""
        if not hasattr(self._adapter, "ainvoke"):
            raise TypeError("adapter does not support async invocation")
        case_list = _validate_cases(cases)
        trial_count = _validate_trials(trials, self._trials)
        identifier = run_id or uuid4().hex
        started = self._clock.now()
        results = []
        for case in case_list:
            for index in range(trial_count):
                results.append(await self._run_trial_async(identifier, case, index))
        finished = self._clock.now()
        return EvalRun(
            run_id=identifier,
            trials=tuple(results),
            started_unix_ns=started.unix_time_ns,
            finished_unix_ns=finished.unix_time_ns,
            metadata=_run_metadata(self._capture, trial_count, metadata),
            environment=_environment(environment),
        )

    def _new_session(self) -> RecordingSession:
        if self._session_factory is not None:
            return self._session_factory()
        store = self._store_factory() if self._store_factory is not None else None
        return RecordingSession(
            store=store,
            capture=self._capture,
            redaction=self._redaction,
            limits=self._limits,
        )

    def _run_trial(self, run_id: str, case: EvalCase, trial_index: int) -> EvalTrial:
        started = self._clock.now()
        session = self._new_session()
        turns: list[EvalTurnResult] = []
        all_bundles = []
        status = EvalExecutionStatus.COMPLETED
        error: EvalError | None = None
        skipped_from: int | None = None
        try:
            for index, turn in enumerate(case.turns):
                if skipped_from is not None:
                    turns.append(_skipped_turn(index, turn))
                    continue
                before = _trace_ids(session)
                turn_started = self._clock.now()
                output: EvalOutput | None = None
                turn_error: EvalError | None = None
                try:
                    output = self._adapter.invoke(
                        EvalInvocation(
                            run_id=run_id,
                            case=case,
                            turn=turn,
                            turn_index=index,
                            trial_index=trial_index,
                            session=session,
                        )
                    )
                    if not isinstance(output, EvalOutput):
                        raise TypeError("adapter must return EvalOutput")
                except Exception as exc:
                    status = _classify_error(exc)
                    turn_error = self._error(exc)
                    error = turn_error
                turn_finished = self._clock.now()
                bundles = tuple(
                    bundle for bundle in session.snapshot() if bundle.trace.trace_id not in before
                )
                all_bundles.extend(bundles)
                metrics = measure_traces(
                    bundles,
                    elapsed_ns=_elapsed(turn_started, turn_finished),
                    diagnostics=session.diagnostics,
                )
                assessments = (
                    ()
                    if turn_error is not None
                    else _evaluate_checks(
                        turn.checks,
                        case,
                        trial_index,
                        turn,
                        index,
                        output,
                        metrics,
                        bundles,
                        tuple(turns),
                    )
                )
                turns.append(
                    EvalTurnResult(
                        index=index,
                        input=turn.payload(),
                        execution_status=(
                            EvalExecutionStatus.COMPLETED if turn_error is None else status
                        ),
                        output=output,
                        assessments=assessments,
                        metrics=metrics,
                        trace_ids=_trace_ids_from(bundles),
                        error=turn_error,
                        started_unix_ns=turn_started.unix_time_ns,
                        finished_unix_ns=turn_finished.unix_time_ns,
                        duration_ns=_elapsed(turn_started, turn_finished),
                    )
                )
                if turn_error is not None:
                    skipped_from = index + 1

            conversation: tuple[EvalAssessment, ...] = ()
            if status is EvalExecutionStatus.COMPLETED:
                conversation = _evaluate_checks(
                    case.checks,
                    case,
                    trial_index,
                    None,
                    None,
                    None,
                    aggregate_turn_metrics(tuple(turn.metrics for turn in turns)),
                    tuple(all_bundles),
                    tuple(turns),
                )
        except Exception as exc:
            status = EvalExecutionStatus.RUNNER_ERROR
            error = self._error(exc)
            conversation = ()
        finally:
            try:
                closed = session.close()
            except Exception:
                closed = False
            if not closed and status is EvalExecutionStatus.COMPLETED:
                status = EvalExecutionStatus.INCOMPLETE
                error = EvalError(exception_type="RecordingSession.close")
        finished = self._clock.now()
        return EvalTrial(
            case_id=case.case_id,
            trial_index=trial_index,
            execution_status=status,
            turns=tuple(turns),
            assessments=conversation,
            metrics=aggregate_turn_metrics(tuple(turn.metrics for turn in turns)),
            error=error,
            started_unix_ns=started.unix_time_ns,
            finished_unix_ns=finished.unix_time_ns,
            duration_ns=_elapsed(started, finished),
        )

    async def _run_trial_async(self, run_id: str, case: EvalCase, trial_index: int) -> EvalTrial:
        started = self._clock.now()
        session = self._new_session()
        turns: list[EvalTurnResult] = []
        all_bundles = []
        status = EvalExecutionStatus.COMPLETED
        error: EvalError | None = None
        conversation: tuple[EvalAssessment, ...] = ()
        skipped_from: int | None = None
        try:
            for index, turn in enumerate(case.turns):
                if skipped_from is not None:
                    turns.append(_skipped_turn(index, turn))
                    continue
                before = _trace_ids(session)
                turn_started = self._clock.now()
                output: EvalOutput | None = None
                turn_error: EvalError | None = None
                try:
                    adapter = cast(Any, self._adapter)
                    output = await adapter.ainvoke(
                        EvalInvocation(
                            run_id=run_id,
                            case=case,
                            turn=turn,
                            turn_index=index,
                            trial_index=trial_index,
                            session=session,
                        )
                    )
                    if not isinstance(output, EvalOutput):
                        raise TypeError("adapter must return EvalOutput")
                except Exception as exc:
                    status = _classify_error(exc)
                    turn_error = self._error(exc)
                    error = turn_error
                turn_finished = self._clock.now()
                bundles = tuple(
                    bundle for bundle in session.snapshot() if bundle.trace.trace_id not in before
                )
                all_bundles.extend(bundles)
                metrics = measure_traces(
                    bundles,
                    elapsed_ns=_elapsed(turn_started, turn_finished),
                    diagnostics=session.diagnostics,
                )
                assessments = (
                    ()
                    if turn_error is not None
                    else _evaluate_checks(
                        turn.checks,
                        case,
                        trial_index,
                        turn,
                        index,
                        output,
                        metrics,
                        bundles,
                        tuple(turns),
                    )
                )
                turns.append(
                    EvalTurnResult(
                        index=index,
                        input=turn.payload(),
                        execution_status=(
                            EvalExecutionStatus.COMPLETED if turn_error is None else status
                        ),
                        output=output,
                        assessments=assessments,
                        metrics=metrics,
                        trace_ids=_trace_ids_from(bundles),
                        error=turn_error,
                        started_unix_ns=turn_started.unix_time_ns,
                        finished_unix_ns=turn_finished.unix_time_ns,
                        duration_ns=_elapsed(turn_started, turn_finished),
                    )
                )
                if turn_error is not None:
                    skipped_from = index + 1

            if status is EvalExecutionStatus.COMPLETED:
                conversation = _evaluate_checks(
                    case.checks,
                    case,
                    trial_index,
                    None,
                    None,
                    None,
                    aggregate_turn_metrics(tuple(turn.metrics for turn in turns)),
                    tuple(all_bundles),
                    tuple(turns),
                )
        except Exception as exc:
            status = EvalExecutionStatus.RUNNER_ERROR
            error = self._error(exc)
            conversation = ()
        finally:
            try:
                closed = session.close()
            except Exception:
                closed = False
            if not closed and status is EvalExecutionStatus.COMPLETED:
                status = EvalExecutionStatus.INCOMPLETE
                error = EvalError(exception_type="RecordingSession.close")
        finished = self._clock.now()
        return EvalTrial(
            case_id=case.case_id,
            trial_index=trial_index,
            execution_status=status,
            turns=tuple(turns),
            assessments=conversation,
            metrics=aggregate_turn_metrics(tuple(turn.metrics for turn in turns)),
            error=error,
            started_unix_ns=started.unix_time_ns,
            finished_unix_ns=finished.unix_time_ns,
            duration_ns=_elapsed(started, finished),
        )

    def _error(self, exc: Exception) -> EvalError:
        exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        message = None
        if self._error_messages:
            try:
                text = str(exc)
            except Exception:
                text = ""
            message = text[:_MAX_ERROR_MESSAGE] or None
        return EvalError(exception_type=exception_type, message=message)


def _validate_cases(cases: Sequence[EvalCase]) -> tuple[EvalCase, ...]:
    if isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of EvalCase records")
    result = tuple(cases)
    if not result:
        raise ValueError("at least one evaluation case is required")
    seen: set[str] = set()
    for case in result:
        if not isinstance(case, EvalCase):
            raise TypeError("cases must contain EvalCase records")
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id: {case.case_id!r}")
        seen.add(case.case_id)
    return result


def _validate_trials(trials: int | None, default: int) -> int:
    value = default if trials is None else trials
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("trials must be a positive integer")
    return value


def _evaluate_checks(
    checks: Sequence[object],
    case: EvalCase,
    trial_index: int,
    turn: EvalTurn | None,
    turn_index: int | None,
    output: EvalOutput | None,
    metrics: EvalMetrics,
    bundles: tuple[TraceBundle, ...],
    prior_turns: tuple[EvalTurnResult, ...],
) -> tuple[EvalAssessment, ...]:
    from .evaluators import EvalContext

    context = EvalContext(
        case=case,
        trial_index=trial_index,
        turn=turn,
        turn_index=turn_index,
        output=output,
        metrics=metrics,
        traces=bundles,
        prior_turns=prior_turns,
    )
    assessments: list[EvalAssessment] = []
    for check in checks:
        name = str(getattr(check, "name", type(check).__name__))
        try:
            assessment = cast(Any, check).evaluate(context)
        except Exception as exc:
            assessments.append(
                EvalAssessment(
                    evaluator=name,
                    status=EvalStatus.ERROR,
                    message=f"evaluator raised {type(exc).__name__}",
                )
            )
            continue
        if not isinstance(assessment, EvalAssessment):
            assessments.append(
                EvalAssessment(
                    evaluator=name,
                    status=EvalStatus.ERROR,
                    message="evaluator did not return an EvalAssessment",
                )
            )
            continue
        assessments.append(assessment)
    return tuple(assessments)


def _skipped_turn(index: int, turn: EvalTurn) -> EvalTurnResult:
    return EvalTurnResult(
        index=index,
        input=turn.payload(),
        execution_status=EvalExecutionStatus.SKIPPED,
    )


def _trace_ids(session: RecordingSession) -> set[Any]:
    return {bundle.trace.trace_id for bundle in session.snapshot()}


def _trace_ids_from(bundles: tuple[TraceBundle, ...]) -> tuple[str, ...]:
    return tuple(str(bundle.trace.trace_id) for bundle in bundles)


def _elapsed(start: ClockReading, end: ClockReading) -> int:
    return max(0, end.monotonic_time_ns - start.monotonic_time_ns)


def _classify_error(exc: Exception) -> EvalExecutionStatus:
    if isinstance(exc, TimeoutError):
        return EvalExecutionStatus.TIMEOUT
    name = type(exc).__name__.casefold()
    if "interrupt" in name:
        return EvalExecutionStatus.INTERRUPTED
    return EvalExecutionStatus.AGENT_ERROR


def _run_metadata(
    capture: CapturePolicy,
    trials: int,
    supplied: Mapping[str, object] | None,
) -> Mapping[str, object]:
    metadata = dict(supplied or {})
    metadata.setdefault("eval.trials", trials)
    metadata.setdefault("eval.capture_mode", capture.mode.value)
    metadata.setdefault("eval.runner", "local")
    return metadata


def _environment(supplied: Mapping[str, object] | None) -> Mapping[str, object]:
    packages: dict[str, str] = {}
    for name in (
        "langgraph-instrumentation",
        "langchain-core",
        "langgraph",
        "deepagents",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }
    if supplied:
        environment["supplied"] = dict(supplied)
    return environment
