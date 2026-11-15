"""Candidate evaluation proposals and feedback records.

Proposals carry rationale, evidence references and fully specified candidate
cases. They are validated structurally, reviewed explicitly, and only then
applied to a golden dataset. Feedback records link observations to a case
result, turn or trace with provenance. Neither type captures application
payloads implicitly: every string field is caller-supplied and bounded.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..models import JSONValue
from .datasets import case_from_dict, case_to_dict, checks_to_specs
from .models import EvalCase

_PROPOSAL_KIND = "langgraph_instrumentation.eval_proposal"
_FEEDBACK_KIND = "langgraph_instrumentation.feedback_record"
_SCHEMA_VERSION = "1"

_MAX_TEXT = 2_000
_MAX_EVIDENCE = 50
_MAX_PROPOSED_CASES = 100


class ProposalStatus(StrEnum):
    """Review state of a candidate evaluation; golden cases stay separate."""

    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvidenceKind(StrEnum):
    """What an evidence reference points at."""

    RUN = "run"
    TRACE = "trace"
    CASE = "case"
    DATASET = "dataset"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer to supporting evidence, never the evidence payload itself."""

    kind: EvidenceKind
    ref: str
    note: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = EvidenceKind(self.kind)
        except ValueError as exc:
            raise ValueError("evidence kind must be one of run, trace, case or dataset") from exc
        object.__setattr__(self, "kind", kind)
        _require_text(self.ref, "evidence ref", limit=256)
        if self.note is not None:
            _require_text(self.note, "evidence note", limit=_MAX_TEXT, allow_empty=True)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"kind": self.kind.value, "ref": self.ref, "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceRef:
        kind = data.get("kind")
        ref = data.get("ref")
        note = data.get("note")
        if not isinstance(kind, str):
            raise TypeError("evidence kind must be a string")
        if not isinstance(ref, str):
            raise TypeError("evidence ref must be a string")
        if note is not None and not isinstance(note, str):
            raise TypeError("evidence note must be a string or null")
        return cls(kind=EvidenceKind(kind), ref=ref, note=note)


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """An observation linked to a case result, turn or trace, with provenance.

    The privacy boundary is structural: content is explicitly supplied by the
    caller, bounded, and never captured from instrumentation payloads. Origin
    distinguishes human review from agent diagnosis.
    """

    feedback_id: str
    author: str
    origin: str
    content: str
    target_case_id: str | None = None
    target_trial_index: int | None = None
    target_turn_index: int | None = None
    target_trace_id: str | None = None
    created_unix_ns: int = 0
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported feedback version: {self.schema_version!r}")
        _require_text(self.feedback_id, "feedback_id", limit=128)
        _require_text(self.author, "author", limit=128)
        _require_text(self.origin, "origin", limit=128)
        _require_text(self.content, "content", limit=_MAX_TEXT)
        if self.target_case_id is None and self.target_trace_id is None:
            raise ValueError("feedback must link to a case or a trace")
        if self.target_case_id is not None:
            _require_text(self.target_case_id, "target case_id", limit=128)
        if self.target_trace_id is not None:
            _require_text(self.target_trace_id, "target trace_id", limit=128)
        for name, value in (
            ("target_trial_index", self.target_trial_index),
            ("target_turn_index", self.target_turn_index),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if isinstance(self.created_unix_ns, bool) or not isinstance(self.created_unix_ns, int):
            raise TypeError("created_unix_ns must be an integer")
        if self.created_unix_ns < 0:
            raise ValueError("created_unix_ns must be non-negative")

    @property
    def kind(self) -> str:
        return _FEEDBACK_KIND

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _FEEDBACK_KIND,
            "feedback_id": self.feedback_id,
            "author": self.author,
            "origin": self.origin,
            "content": self.content,
            "target_case_id": self.target_case_id,
            "target_trial_index": self.target_trial_index,
            "target_turn_index": self.target_turn_index,
            "target_trace_id": self.target_trace_id,
            "created_unix_ns": self.created_unix_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FeedbackRecord:
        if data.get("kind") != _FEEDBACK_KIND:
            raise ValueError("not a feedback record")
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported feedback version: {version!r}")
        return cls(
            feedback_id=_req_string(data, "feedback_id"),
            author=_req_string(data, "author"),
            origin=_req_string(data, "origin"),
            content=_req_string(data, "content"),
            target_case_id=_opt_string(data, "target_case_id"),
            target_trial_index=_opt_int(data, "target_trial_index"),
            target_turn_index=_opt_int(data, "target_turn_index"),
            target_trace_id=_opt_string(data, "target_trace_id"),
            created_unix_ns=_req_int(data, "created_unix_ns"),
            schema_version=str(version),
        )


@dataclass(frozen=True, slots=True)
class ProposalValidation:
    """Structural validation outcome for a candidate proposal."""

    passed: bool
    findings: tuple[str, ...] = ()
    checked_cases: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        object.__setattr__(self, "findings", tuple(self.findings))
        if isinstance(self.checked_cases, bool) or not isinstance(self.checked_cases, int):
            raise TypeError("checked_cases must be an integer")
        if self.checked_cases < 0:
            raise ValueError("checked_cases must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "passed": self.passed,
            "findings": list(self.findings),
            "checked_cases": self.checked_cases,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ProposalValidation:
        passed = data.get("passed")
        findings = data.get("findings")
        checked = data.get("checked_cases")
        if not isinstance(passed, bool):
            raise TypeError("passed must be a boolean")
        if not isinstance(findings, list):
            raise TypeError("findings must be a list")
        if isinstance(checked, bool) or not isinstance(checked, int):
            raise TypeError("checked_cases must be an integer")
        return cls(
            passed=passed,
            findings=tuple(str(item) for item in findings),
            checked_cases=checked,
        )


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """An explicit accept/reject decision with provenance."""

    reviewer: str
    decision: ProposalStatus
    note: str | None = None
    reviewed_unix_ns: int = 0

    def __post_init__(self) -> None:
        _require_text(self.reviewer, "reviewer", limit=128)
        try:
            decision = ProposalStatus(self.decision)
        except ValueError as exc:
            raise ValueError("decision must be accepted or rejected") from exc
        object.__setattr__(self, "decision", decision)
        if self.note is not None:
            _require_text(self.note, "review note", limit=_MAX_TEXT, allow_empty=True)
        if isinstance(self.reviewed_unix_ns, bool) or not isinstance(self.reviewed_unix_ns, int):
            raise TypeError("reviewed_unix_ns must be an integer")
        if self.reviewed_unix_ns < 0:
            raise ValueError("reviewed_unix_ns must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "reviewer": self.reviewer,
            "decision": self.decision.value,
            "note": self.note,
            "reviewed_unix_ns": self.reviewed_unix_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReviewRecord:
        reviewer = data.get("reviewer")
        decision = data.get("decision")
        note = data.get("note")
        reviewed = data.get("reviewed_unix_ns")
        if not isinstance(reviewer, str):
            raise TypeError("reviewer must be a string")
        if not isinstance(decision, str):
            raise TypeError("decision must be a string")
        if note is not None and not isinstance(note, str):
            raise TypeError("review note must be a string or null")
        if isinstance(reviewed, bool) or not isinstance(reviewed, int):
            raise TypeError("reviewed_unix_ns must be an integer")
        return cls(
            reviewer=reviewer,
            decision=ProposalStatus(decision),
            note=note,
            reviewed_unix_ns=reviewed,
        )


@dataclass(frozen=True, slots=True)
class EvalProposal:
    """A candidate evaluation change, distinct from accepted golden cases."""

    proposal_id: str
    author: str
    rationale: str
    proposed_cases: tuple[EvalCase, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    status: ProposalStatus = ProposalStatus.CANDIDATE
    validation: ProposalValidation | None = None
    review: ReviewRecord | None = None
    created_unix_ns: int = 0
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported proposal version: {self.schema_version!r}")
        _require_text(self.proposal_id, "proposal_id", limit=128)
        _require_text(self.author, "author", limit=128)
        _require_text(self.rationale, "rationale", limit=_MAX_TEXT)
        cases = tuple(self.proposed_cases)
        if not cases:
            raise ValueError("a proposal must contain at least one case")
        if len(cases) > _MAX_PROPOSED_CASES:
            raise ValueError(f"a proposal holds at most {_MAX_PROPOSED_CASES} cases")
        for case in cases:
            if not isinstance(case, EvalCase):
                raise TypeError("proposed cases must be EvalCase records")
        evidence = tuple(self.evidence)
        if len(evidence) > _MAX_EVIDENCE:
            raise ValueError(f"a proposal holds at most {_MAX_EVIDENCE} evidence refs")
        for ref in evidence:
            if not isinstance(ref, EvidenceRef):
                raise TypeError("evidence must be EvidenceRef records")
        try:
            status = ProposalStatus(self.status)
        except ValueError as exc:
            raise ValueError("status must be candidate, accepted or rejected") from exc
        if self.validation is not None and not isinstance(self.validation, ProposalValidation):
            raise TypeError("validation must be ProposalValidation or null")
        if self.review is not None and not isinstance(self.review, ReviewRecord):
            raise TypeError("review must be ReviewRecord or null")
        if (self.status is ProposalStatus.CANDIDATE) == (self.review is not None):
            if self.status is ProposalStatus.CANDIDATE and self.review is not None:
                raise ValueError("a candidate proposal must not carry a review")
            if self.status is not ProposalStatus.CANDIDATE and self.review is None:
                raise ValueError("a decided proposal must carry its review")
        if self.review is not None and self.review.decision.value != self.status.value:
            raise ValueError("review decision must match proposal status")
        if isinstance(self.created_unix_ns, bool) or not isinstance(self.created_unix_ns, int):
            raise TypeError("created_unix_ns must be an integer")
        if self.created_unix_ns < 0:
            raise ValueError("created_unix_ns must be non-negative")
        object.__setattr__(self, "proposed_cases", cases)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kind": _PROPOSAL_KIND,
            "proposal_id": self.proposal_id,
            "author": self.author,
            "rationale": self.rationale,
            "status": self.status.value,
            "proposed_cases": [case_to_dict(case) for case in self.proposed_cases],
            "evidence": [ref.to_dict() for ref in self.evidence],
            "validation": self.validation.to_dict() if self.validation else None,
            "review": self.review.to_dict() if self.review else None,
            "created_unix_ns": self.created_unix_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvalProposal:
        if data.get("kind") != _PROPOSAL_KIND:
            raise ValueError("not an evaluation proposal")
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported proposal version: {version!r}")
        proposed = data.get("proposed_cases")
        evidence = data.get("evidence", [])
        validation = data.get("validation")
        review = data.get("review")
        if not isinstance(proposed, list) or not proposed:
            raise ValueError("a proposal must contain at least one case")
        if not isinstance(evidence, list):
            raise TypeError("evidence must be a list")
        return cls(
            proposal_id=_req_string(data, "proposal_id"),
            author=_req_string(data, "author"),
            rationale=_req_string(data, "rationale"),
            proposed_cases=tuple(
                case_from_dict(_mapping(item, "proposed case")) for item in proposed
            ),
            evidence=tuple(
                EvidenceRef.from_dict(_mapping(item, "evidence ref")) for item in evidence
            ),
            status=ProposalStatus(_req_string(data, "status")),
            validation=(
                None
                if validation is None
                else ProposalValidation.from_dict(_mapping(validation, "validation"))
            ),
            review=(None if review is None else ReviewRecord.from_dict(_mapping(review, "review"))),
            created_unix_ns=_req_int(data, "created_unix_ns"),
            schema_version=str(version),
        )


def propose(
    cases: Sequence[EvalCase],
    *,
    proposal_id: str | None = None,
    author: str = "unknown",
    rationale: str,
    evidence: Sequence[EvidenceRef] = (),
    created_unix_ns: int | None = None,
    validate_against: Sequence[EvalCase] = (),
) -> EvalProposal:
    """Create a candidate proposal and record its structural validation.

    Validation is structural only: portable checks, well-formed evidence and
    ID collisions against the base dataset. It never executes an agent.
    """
    items = tuple(cases)
    refs = tuple(evidence)
    base = tuple(validate_against)
    for case in base:
        if not isinstance(case, EvalCase):
            raise TypeError("base cases must be EvalCase records")
    proposal = EvalProposal(
        proposal_id=proposal_id or f"proposal-{time.time_ns()}",
        author=author,
        rationale=rationale,
        proposed_cases=items,
        evidence=refs,
        created_unix_ns=created_unix_ns if created_unix_ns is not None else time.time_ns(),
    )
    validation = validate_proposal(proposal, base)
    return EvalProposal(
        proposal_id=proposal.proposal_id,
        author=proposal.author,
        rationale=proposal.rationale,
        proposed_cases=proposal.proposed_cases,
        evidence=proposal.evidence,
        status=ProposalStatus.CANDIDATE,
        validation=validation,
        created_unix_ns=proposal.created_unix_ns,
        schema_version=proposal.schema_version,
    )


def validate_proposal(
    proposal: EvalProposal, base_cases: Sequence[EvalCase] = ()
) -> ProposalValidation:
    """Validate a proposal structurally without executing anything."""
    if not isinstance(proposal, EvalProposal):
        raise TypeError("proposal must be an EvalProposal")
    base = tuple(base_cases)
    for case in base:
        if not isinstance(case, EvalCase):
            raise TypeError("base cases must be EvalCase records")
    findings: list[str] = []
    base_ids = {case.case_id for case in base}
    for case in proposal.proposed_cases:
        try:
            checks_to_specs(case.checks)
            for turn in case.turns:
                checks_to_specs(turn.checks)
        except ValueError as exc:
            findings.append(f"case {case.case_id!r}: {exc}")
        if case.case_id in base_ids:
            findings.append(f"case {case.case_id!r} already exists in the base dataset")
    for ref in proposal.evidence:
        if not ref.ref.strip():
            findings.append("evidence ref must not be empty")
    return ProposalValidation(
        passed=not findings, findings=tuple(findings), checked_cases=len(proposal.proposed_cases)
    )


def review_proposal(
    proposal: EvalProposal,
    *,
    reviewer: str,
    decision: str | ProposalStatus,
    note: str | None = None,
    reviewed_unix_ns: int | None = None,
) -> EvalProposal:
    """Record an explicit accept/reject decision with provenance.

    Write access is opt-in: reviewer and decision are required arguments, and
    the input proposal is never mutated.
    """
    if not isinstance(proposal, EvalProposal):
        raise TypeError("proposal must be an EvalProposal")
    if proposal.status is not ProposalStatus.CANDIDATE:
        raise ValueError("only a candidate proposal can be reviewed")
    try:
        status = ProposalStatus(decision)
    except ValueError as exc:
        raise ValueError("decision must be accepted or rejected") from exc
    if status is ProposalStatus.CANDIDATE:
        raise ValueError("decision must be accepted or rejected")
    review = ReviewRecord(
        reviewer=reviewer,
        decision=status,
        note=note,
        reviewed_unix_ns=reviewed_unix_ns if reviewed_unix_ns is not None else time.time_ns(),
    )
    return EvalProposal(
        proposal_id=proposal.proposal_id,
        author=proposal.author,
        rationale=proposal.rationale,
        proposed_cases=proposal.proposed_cases,
        evidence=proposal.evidence,
        status=status,
        validation=proposal.validation,
        review=review,
        created_unix_ns=proposal.created_unix_ns,
        schema_version=proposal.schema_version,
    )


def apply_proposal(
    base_cases: Sequence[EvalCase],
    proposal: EvalProposal,
    *,
    replace: bool = False,
) -> tuple[EvalCase, ...]:
    """Incorporate an accepted proposal into golden cases.

    Without ``replace``, an ID collision fails instead of silently overwriting
    a golden case. The base tuple is never mutated.
    """
    base = tuple(base_cases)
    for case in base:
        if not isinstance(case, EvalCase):
            raise TypeError("base cases must be EvalCase records")
    if not isinstance(proposal, EvalProposal):
        raise TypeError("proposal must be an EvalProposal")
    if proposal.status is not ProposalStatus.ACCEPTED:
        raise ValueError("only an accepted proposal can be applied")
    existing = {case.case_id for case in base}
    incoming = [case for case in proposal.proposed_cases]
    collisions = [case.case_id for case in incoming if case.case_id in existing]
    if collisions and not replace:
        raise ValueError(
            "proposal collides with golden case(s) "
            + ", ".join(sorted(collisions))
            + "; pass replace=True to revise them explicitly"
        )
    if not replace:
        return (*base, *incoming)
    replaced = {case.case_id: case for case in incoming}
    return tuple(replaced.get(case.case_id, case) for case in base) + tuple(
        case for case in incoming if case.case_id not in existing
    )


def _require_text(value: object, name: str, *, limit: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _req_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _opt_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _req_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _opt_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null")
    return value
