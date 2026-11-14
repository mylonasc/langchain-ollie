"""Offline tests for feedback records and the proposal lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_instrumentation import (
    EvalCase,
    EvalTurn,
    ExactText,
    FunctionEvaluator,
    load_cases,
    save_cases,
)
from langgraph_instrumentation.evals.proposals import (
    EvalProposal,
    EvidenceKind,
    EvidenceRef,
    FeedbackRecord,
    ProposalStatus,
    apply_proposal,
    propose,
    review_proposal,
)


def _case(case_id: str = "new-case") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        turns=(EvalTurn.user("Summarize the outage", checks=(ExactText("ok"),)),),
    )


def _feedback(**overrides: Any) -> FeedbackRecord:
    values: dict[str, Any] = {
        "feedback_id": "fb-1",
        "author": "sme",
        "origin": "human-review",
        "content": "The agent missed the outage window.",
        "target_case_id": "outage",
        "target_turn_index": 0,
    }
    values.update(overrides)
    return FeedbackRecord(**values)


def _proposal(**overrides: Any) -> EvalProposal:
    values: dict[str, Any] = {
        "proposal_id": "p-1",
        "author": "agent",
        "rationale": "Cover the missed outage window from run r1.",
        "proposed_cases": (_case(),),
        "evidence": (
            EvidenceRef(kind=EvidenceKind.RUN, ref="r1"),
            EvidenceRef(kind=EvidenceKind.TRACE, ref="abc123"),
        ),
        "created_unix_ns": 1,
    }
    values.update(overrides)
    return EvalProposal(**values)


def test_feedback_requires_a_target_and_bounds_content() -> None:
    record = _feedback()
    assert record.to_dict()["kind"] == "langgraph_instrumentation.feedback_record"
    assert FeedbackRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="must link to a case or a trace"):
        _feedback(target_case_id=None)
    with pytest.raises(ValueError, match="at most"):
        _feedback(content="x" * 2001)
    trace_only = _feedback(target_case_id=None, target_trace_id="deadbeef")
    assert trace_only.target_trace_id == "deadbeef"


def test_propose_validates_without_executing(tmp_path) -> None:
    base = (_case("golden"),)
    proposal = propose(
        [_case()],
        author="agent",
        rationale="Add coverage for the outage window.",
        evidence=[EvidenceRef(kind=EvidenceKind.RUN, ref="r1")],
        created_unix_ns=7,
        validate_against=base,
    )
    assert proposal.status is ProposalStatus.CANDIDATE
    assert proposal.review is None
    assert proposal.validation is not None
    assert proposal.validation.passed is True
    assert proposal.validation.checked_cases == 1
    assert EvalProposal.from_dict(proposal.to_dict()) == proposal


def test_proposal_flags_collisions_and_unportable_checks() -> None:
    custom = EvalCase(
        case_id="custom",
        turns=(EvalTurn.user("x", checks=(FunctionEvaluator("custom", lambda context: True),)),),
    )
    proposal = propose(
        [custom, _case("golden")],
        author="agent",
        rationale="Try custom and colliding cases.",
        validate_against=[_case("golden")],
    )
    assert proposal.validation is not None
    assert proposal.validation.passed is False
    findings = "\n".join(proposal.validation.findings)
    assert "golden" in findings
    assert "custom" in findings


def test_review_is_explicit_and_recorded_with_provenance() -> None:
    proposal = _proposal()
    with pytest.raises(ValueError, match="reviewer"):
        review_proposal(proposal, reviewer="", decision=ProposalStatus.ACCEPTED)
    with pytest.raises(ValueError, match="candidate"):
        review_proposal(
            review_proposal(
                proposal,
                reviewer="sme",
                decision=ProposalStatus.ACCEPTED,
                reviewed_unix_ns=9,
            ),
            reviewer="sme",
            decision=ProposalStatus.REJECTED,
        )
    accepted = review_proposal(
        proposal,
        reviewer="sme",
        decision="accepted",
        note="Reproduces the outage window.",
        reviewed_unix_ns=9,
    )
    assert accepted.status is ProposalStatus.ACCEPTED
    assert accepted.review is not None
    assert accepted.review.reviewer == "sme"
    assert accepted.review.reviewed_unix_ns == 9
    assert accepted.validation == proposal.validation
    assert proposal.status is ProposalStatus.CANDIDATE
    assert EvalProposal.from_dict(accepted.to_dict()) == accepted

    rejected = review_proposal(proposal, reviewer="sme", decision="rejected")
    assert rejected.status is ProposalStatus.REJECTED


def test_apply_requires_acceptance_and_protects_golden_cases() -> None:
    base = [_case("golden")]
    proposal = _proposal()
    with pytest.raises(ValueError, match="only an accepted proposal"):
        apply_proposal(base, proposal)

    accepted = review_proposal(proposal, reviewer="sme", decision="accepted")
    merged = apply_proposal(base, accepted)
    assert [case.case_id for case in merged] == ["golden", "new-case"]
    assert [case.case_id for case in base] == ["golden"]

    colliding = _proposal(
        proposed_cases=(
            EvalCase(
                case_id="golden",
                turns=(EvalTurn.user("A revised outage question", checks=(ExactText("ok"),)),),
            ),
        )
    )
    accepted_colliding = review_proposal(colliding, reviewer="sme", decision="accepted")
    with pytest.raises(ValueError, match="collides with golden"):
        apply_proposal(base, accepted_colliding)
    revised = apply_proposal(base, accepted_colliding, replace=True)
    assert [case.case_id for case in revised] == ["golden"]
    assert revised[0].turns[0].messages != base[0].turns[0].messages


def test_proposal_and_feedback_persist_as_versioned_artifacts(tmp_path) -> None:
    from langgraph_instrumentation import (
        load_feedback,
        load_proposal,
        save_feedback,
        save_proposal,
    )

    proposal_path, feedback_path = tmp_path / "p.json", tmp_path / "fb.json"
    save_proposal(_proposal(), proposal_path)
    save_feedback(_feedback(), feedback_path)
    assert load_proposal(proposal_path) == _proposal()
    assert load_feedback(feedback_path) == _feedback()

    broken = proposal_path.read_text(encoding="utf-8").replace(
        '"schema_version": "1"', '"schema_version": "99"'
    )
    proposal_path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported proposal version|invalid"):
        load_proposal(proposal_path)


def test_accepted_cases_save_as_golden_datasets(tmp_path) -> None:
    proposal = review_proposal(_proposal(), reviewer="sme", decision="accepted")
    golden = apply_proposal([], proposal)
    destination = tmp_path / "golden.json"
    save_cases(golden, destination)
    assert [case.case_id for case in load_cases(destination)] == ["new-case"]
