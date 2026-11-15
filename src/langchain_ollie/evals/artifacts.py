"""Versioned, portable persistence for evaluation artifacts.

Runs, trial aggregates and baseline comparisons serialize as canonical JSON
(single artifact per ``.json`` file, one artifact per line for ``.jsonl``).
Full trace payloads are never embedded; artifacts carry trace references only.
Byte-stability: saving a loaded artifact reproduces the source bytes exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .aggregation import RunAggregate
from .comparison import ComparisonSummary
from .models import EvalRun
from .proposals import EvalProposal, FeedbackRecord

_RUN_KIND = "langgraph_instrumentation.eval_run"
_AGGREGATE_KIND = "langgraph_instrumentation.eval_aggregate"
_COMPARISON_KIND = "langgraph_instrumentation.eval_comparison"
_PROPOSAL_KIND = "langgraph_instrumentation.eval_proposal"
_FEEDBACK_KIND = "langgraph_instrumentation.feedback_record"

Artifact = EvalRun | RunAggregate | ComparisonSummary | EvalProposal | FeedbackRecord


def save_run(run: EvalRun, path: str | Path) -> Path:
    """Save one evaluation run as canonical JSON."""
    if not isinstance(run, EvalRun):
        raise TypeError("run must be an EvalRun")
    return _write_json(path, run.to_dict())


def load_run(path: str | Path) -> EvalRun:
    """Load one evaluation run; kind and version mismatches fail explicitly."""
    source = Path(path)
    payload = _read_json(source, EvalRun)
    try:
        return EvalRun.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation run in {source}: {exc}") from exc


def save_aggregate(aggregate: RunAggregate, path: str | Path) -> Path:
    """Save one trial aggregate as canonical JSON."""
    if not isinstance(aggregate, RunAggregate):
        raise TypeError("aggregate must be a RunAggregate")
    return _write_json(path, aggregate.to_dict())


def load_aggregate(path: str | Path) -> RunAggregate:
    """Load one trial aggregate; kind and version mismatches fail explicitly."""
    source = Path(path)
    payload = _read_json(source, RunAggregate)
    try:
        return RunAggregate.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid trial aggregate in {source}: {exc}") from exc


def save_comparison(comparison: ComparisonSummary, path: str | Path) -> Path:
    """Save one baseline comparison as canonical JSON."""
    if not isinstance(comparison, ComparisonSummary):
        raise TypeError("comparison must be a ComparisonSummary")
    return _write_json(path, comparison.to_dict())


def load_comparison(path: str | Path) -> ComparisonSummary:
    """Load one baseline comparison; kind and version mismatches fail explicitly."""
    source = Path(path)
    payload = _read_json(source, ComparisonSummary)
    try:
        return ComparisonSummary.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid baseline comparison in {source}: {exc}") from exc


def save_artifacts(objects: Sequence[Artifact], path: str | Path) -> Path:
    """Save several artifacts as JSONL, one canonical object per line."""
    items = tuple(objects)
    if not items:
        raise ValueError("at least one artifact is required")
    for item in items:
        if not isinstance(
            item, (EvalRun, RunAggregate, ComparisonSummary, EvalProposal, FeedbackRecord)
        ):
            raise TypeError(
                "artifacts must be EvalRun, RunAggregate, ComparisonSummary, "
                "EvalProposal or FeedbackRecord records"
            )
    destination = Path(path)
    if destination.suffix.casefold() != ".jsonl":
        raise ValueError("multi-artifact path must end with .jsonl")
    lines = [json.dumps(item.to_dict(), sort_keys=True) for item in items]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def load_artifacts(path: str | Path) -> tuple[Artifact, ...]:
    """Load JSONL artifacts, dispatching each line by its ``kind`` marker."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read artifact file: {source}: {exc}") from exc
    records = [line for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"artifact file is empty: {source}")
    return tuple(_artifact_from_line(line, source) for line in records)


def load_any(path: str | Path) -> Artifact:
    """Load a single ``.json`` artifact of any evaluation kind."""
    source = Path(path)
    if source.suffix.casefold() != ".json":
        raise ValueError(f"single-artifact path must end with .json: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in artifact file: {source}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read artifact file: {source}: {exc}") from exc
    return _dispatch(payload, source)


def canonical_json(data: Mapping[str, object]) -> str:
    """Render canonical bytes for an artifact mapping."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _write_json(path: str | Path, data: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError(f"single-artifact path must end with .json: {destination}")
    destination.write_text(canonical_json(data), encoding="utf-8")
    return destination


def _read_json(path: str | Path, expected: type) -> Mapping[str, object]:
    source = Path(path)
    if source.suffix.casefold() != ".json":
        raise ValueError(f"single-artifact path must end with .json: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in artifact file: {source}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read artifact file: {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact root must be an object: {source}")
    _check_kind(payload, source, expected)
    return payload


def save_proposal(proposal: EvalProposal, path: str | Path) -> Path:
    """Save one candidate proposal as canonical JSON."""
    if not isinstance(proposal, EvalProposal):
        raise TypeError("proposal must be an EvalProposal")
    return _write_json(path, proposal.to_dict())


def load_proposal(path: str | Path) -> EvalProposal:
    """Load one candidate proposal; kind and version mismatches fail explicitly."""
    source = Path(path)
    payload = _read_json(source, EvalProposal)
    try:
        return EvalProposal.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation proposal in {source}: {exc}") from exc


def save_feedback(feedback: FeedbackRecord, path: str | Path) -> Path:
    """Save one feedback record as canonical JSON."""
    if not isinstance(feedback, FeedbackRecord):
        raise TypeError("feedback must be a FeedbackRecord")
    return _write_json(path, feedback.to_dict())


def load_feedback(path: str | Path) -> FeedbackRecord:
    """Load one feedback record; kind and version mismatches fail explicitly."""
    source = Path(path)
    payload = _read_json(source, FeedbackRecord)
    try:
        return FeedbackRecord.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid feedback record in {source}: {exc}") from exc


def _check_kind(payload: Mapping[str, object], source: Path, expected: type) -> None:
    if expected is EvalRun:
        expected_kind = _RUN_KIND
    elif expected is RunAggregate:
        expected_kind = _AGGREGATE_KIND
    elif expected is ComparisonSummary:
        expected_kind = _COMPARISON_KIND
    elif expected is EvalProposal:
        expected_kind = _PROPOSAL_KIND
    elif expected is FeedbackRecord:
        expected_kind = _FEEDBACK_KIND
    else:
        raise TypeError(f"unknown artifact type: {expected!r}")
    actual = payload.get("kind")
    if actual != expected_kind:
        raise ValueError(f"not a {expected_kind} artifact (found kind {actual!r}): {source}")


def _artifact_from_line(line: str, source: Path) -> Artifact:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL record in {source}: {exc}") from exc
    return _dispatch(payload, source)


def _dispatch(payload: object, source: Path) -> Artifact:
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact record must be an object: {source}")
    kind = payload.get("kind")
    try:
        if kind == _RUN_KIND:
            return EvalRun.from_dict(payload)
        if kind == _AGGREGATE_KIND:
            return RunAggregate.from_dict(payload)
        if kind == _COMPARISON_KIND:
            return ComparisonSummary.from_dict(payload)
        if kind == _PROPOSAL_KIND:
            return EvalProposal.from_dict(payload)
        if kind == _FEEDBACK_KIND:
            return FeedbackRecord.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation artifact in {source}: {exc}") from exc
    raise ValueError(f"unknown evaluation artifact kind {kind!r}: {source}")
