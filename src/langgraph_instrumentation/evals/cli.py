"""Thin CLI over the public evaluation Python API.

Every command delegates to the library API; no execution, aggregation or
comparison logic lives here. Machine-readable JSON goes to stdout by default
(``--format text`` for humans). Exit codes: 0 success or gate passed,
1 gate failed (run failed, regression found, validation finding),
2 usage or artifact error.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import TraceBundle
from ..policies import CapturePolicy
from .aggregation import MetricAggregation, TrialAggregation, aggregate_run
from .artifacts import (
    load_any,
    load_artifacts,
    load_comparison,
    load_proposal,
    load_run,
    save_comparison,
    save_proposal,
    save_run,
)
from .comparison import MetricDirection, MetricRule, compare_runs
from .datasets import load_cases, save_cases
from .models import EvalCase, EvalRun
from .proposals import EvalProposal, EvidenceKind
from .report import render_report
from .runner import EvalRunner

_FORMATS = ("json", "text")
_CAPTURES = ("metadata", "bounded", "full")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation CLI command; return the process exit code."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.function(args)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


class ArtifactError(Exception):
    """A usage or artifact error (exit code 2)."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="execute a dataset against an agent")
    run_cmd.add_argument("dataset", type=Path)
    run_cmd.add_argument("--agent", required=True, help="MODULE:ATTR agent entrypoint")
    run_cmd.add_argument("--out", required=True, type=Path)
    run_cmd.add_argument("--trials", type=int, default=1)
    run_cmd.add_argument("--run-id", default=None)
    run_cmd.add_argument("--capture", choices=_CAPTURES, default="metadata")
    run_cmd.add_argument("--format", choices=_FORMATS, default="json")
    run_cmd.set_defaults(function=_run_suite)

    cases_cmd = commands.add_parser("cases", help="list or inspect dataset cases")
    cases_cmd.add_argument("dataset", type=Path)
    cases_cmd.add_argument("--case", default=None)
    cases_cmd.add_argument("--format", choices=_FORMATS, default="json")
    cases_cmd.set_defaults(function=_list_cases)

    show_cmd = commands.add_parser("show", help="summarize a saved run")
    show_cmd.add_argument("run", type=Path)
    show_cmd.add_argument("--format", choices=_FORMATS, default="json")
    show_cmd.set_defaults(function=_show_run)

    compare_cmd = commands.add_parser("compare", help="compare a candidate run to a baseline")
    compare_cmd.add_argument("baseline", type=Path)
    compare_cmd.add_argument("candidate", type=Path)
    compare_cmd.add_argument(
        "--rule",
        action="append",
        default=[],
        metavar="METRIC:DIRECTION[:ABS][:REL]",
        help="repeatable metric gate",
    )
    compare_cmd.add_argument(
        "--trials-policy",
        choices=[item.value for item in TrialAggregation],
        default=TrialAggregation.WORST.value,
    )
    compare_cmd.add_argument(
        "--metrics-policy",
        choices=[item.value for item in MetricAggregation],
        default=MetricAggregation.WORST.value,
    )
    compare_cmd.add_argument("--allow-added-removed", action="store_true")
    compare_cmd.add_argument("--out", default=None, type=Path)
    compare_cmd.add_argument("--format", choices=_FORMATS, default="json")
    compare_cmd.set_defaults(function=_compare)

    validate_cmd = commands.add_parser("validate", help="validate a dataset or artifact")
    validate_cmd.add_argument("path", type=Path)
    validate_cmd.add_argument("--format", choices=_FORMATS, default="json")
    validate_cmd.set_defaults(function=_validate)

    report_cmd = commands.add_parser("report", help="write a static HTML report")
    report_cmd.add_argument("run", type=Path)
    report_cmd.add_argument("--comparison", default=None, type=Path)
    report_cmd.add_argument("--out", required=True, type=Path)
    report_cmd.add_argument("--title", default="Evaluation report")
    report_cmd.add_argument("--format", choices=_FORMATS, default="json")
    report_cmd.set_defaults(function=_report)

    inspect_cmd = commands.add_parser(
        "inspect", help="bounded read-only failure and turn inspection"
    )
    inspect_cmd.add_argument("run", type=Path)
    inspect_cmd.add_argument("--case", default=None)
    inspect_cmd.add_argument("--turn", default=None, type=int)
    inspect_cmd.add_argument("--max-cases", default=20, type=int)
    inspect_cmd.add_argument("--max-text", default=2000, type=int)
    inspect_cmd.add_argument("--max-calls", default=50, type=int)
    inspect_cmd.add_argument(
        "--traces",
        default=None,
        type=Path,
        help="read-only trace database used only to resolve tool/LLM calls",
    )
    inspect_cmd.add_argument("--format", choices=_FORMATS, default="json")
    inspect_cmd.set_defaults(function=_inspect)

    propose_cmd = commands.add_parser("propose", help="create a candidate evaluation proposal")
    propose_cmd.add_argument("--base-dataset", required=True, type=Path)
    propose_cmd.add_argument("--candidate-cases", required=True, type=Path)
    propose_cmd.add_argument("--rationale", required=True)
    propose_cmd.add_argument("--author", default="unknown")
    propose_cmd.add_argument("--evidence-run", default=None, type=Path)
    propose_cmd.add_argument("--evidence-trace", action="append", default=[])
    propose_cmd.add_argument("--proposal-id", default=None)
    propose_cmd.add_argument("--out", required=True, type=Path)
    propose_cmd.add_argument("--format", choices=_FORMATS, default="json")
    propose_cmd.set_defaults(function=_propose)

    validate_proposal_cmd = commands.add_parser(
        "proposal-validate", help="validate a candidate proposal"
    )
    validate_proposal_cmd.add_argument("proposal", type=Path)
    validate_proposal_cmd.add_argument("--base-dataset", default=None, type=Path)
    validate_proposal_cmd.add_argument("--format", choices=_FORMATS, default="json")
    validate_proposal_cmd.set_defaults(function=_proposal_validate)

    review_cmd = commands.add_parser(
        "proposal-review", help="explicitly accept or reject a proposal"
    )
    review_cmd.add_argument("proposal", type=Path)
    review_cmd.add_argument("--decision", required=True, choices=("accept", "reject"))
    review_cmd.add_argument("--reviewer", required=True)
    review_cmd.add_argument("--note", default=None)
    review_cmd.add_argument("--out", required=True, type=Path)
    review_cmd.add_argument("--format", choices=_FORMATS, default="json")
    review_cmd.set_defaults(function=_proposal_review)

    apply_cmd = commands.add_parser(
        "apply-proposal", help="merge an accepted proposal into golden cases"
    )
    apply_cmd.add_argument("dataset", type=Path)
    apply_cmd.add_argument("proposal", type=Path)
    apply_cmd.add_argument("--out", required=True, type=Path)
    apply_cmd.add_argument("--replace", action="store_true")
    apply_cmd.add_argument("--format", choices=_FORMATS, default="json")
    apply_cmd.set_defaults(function=_apply_proposal)

    return parser


def _emit(document: dict[str, Any], format: str) -> None:
    if format == "json":
        print(json.dumps(document, sort_keys=True))
    else:
        for key, value in document.items():
            print(f"{key}: {_text(value)}")


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


def _run_suite(args: argparse.Namespace) -> int:
    from .adapters import LangGraphEvalAdapter

    cases = _require_cases(args.dataset)
    app = _import_agent(args.agent)
    capture = {
        "metadata": CapturePolicy.metadata(),
        "bounded": CapturePolicy.bounded(),
        "full": CapturePolicy.full(),
    }[args.capture]
    if isinstance(args.trials, bool) or args.trials < 1:
        raise ArtifactError("--trials must be a positive integer")
    run = EvalRunner(LangGraphEvalAdapter(app), capture=capture).run(
        cases, run_id=args.run_id, trials=args.trials
    )
    save_run(run, args.out)
    failed = sorted({trial.case_id for trial in run.trials if not trial.passed})
    _emit(
        {
            "run": str(args.out),
            "run_id": run.run_id,
            "passed": run.passed,
            "trials": len(run.trials),
            "failed_cases": failed,
        },
        args.format,
    )
    return 0 if run.passed else 1


def _list_cases(args: argparse.Namespace) -> int:
    cases = _require_cases(args.dataset)
    if args.case is not None:
        selected = [case for case in cases if case.case_id == args.case]
        if not selected:
            raise ArtifactError(f"unknown case_id: {args.case!r}")
        _emit(_case_document(selected[0]), args.format)
        return 0
    if args.format == "json":
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "cases": [
                        {
                            "case_id": case.case_id,
                            "turns": len(case.turns),
                            "checks": len(case.checks)
                            + sum(len(turn.checks) for turn in case.turns),
                        }
                        for case in cases
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        for case in cases:
            print(f"{case.case_id}: {len(case.turns)} turn(s)")
    return 0


def _show_run(args: argparse.Namespace) -> int:
    run = _require_run(args.run)
    aggregate = aggregate_run(run)
    failed = sorted({trial.case_id for trial in run.trials if not trial.passed})
    _emit(
        {
            "run": str(args.run),
            "run_id": run.run_id,
            "passed": run.passed,
            "cases": [
                {
                    "case_id": case.case_id,
                    "passed": case.passed,
                    "pass_rate": case.pass_rate,
                    "trials": case.trial_count,
                    "execution_failures": case.execution_failures,
                    "assessment_errors": case.assessment_errors,
                }
                for case in aggregate.cases
            ],
            "failed_cases": failed,
        },
        args.format,
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = _require_run(args.baseline)
    candidate = _require_run(args.candidate)
    rules = tuple(_parse_rule(spec) for spec in args.rule)
    summary = compare_runs(
        baseline,
        candidate,
        rules=rules,
        trials=TrialAggregation(args.trials_policy),
        metrics=MetricAggregation(args.metrics_policy),
        require_same_cases=not args.allow_added_removed,
    )
    if args.out is not None:
        save_comparison(summary, args.out)
    if args.format == "json":
        print(json.dumps(summary.to_dict(), sort_keys=True))
    else:
        _emit(
            {
                "passed": summary.passed,
                "regressed": list(summary.regressed_case_ids),
                "improved": list(summary.improved_case_ids),
                "unchanged": list(summary.unchanged_case_ids),
            },
            args.format,
        )
    return 0 if summary.passed else 1


def _validate(args: argparse.Namespace) -> int:
    path = args.path
    if not path.is_file():
        raise ArtifactError(f"file does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix not in (".json", ".jsonl", ".csv"):
        raise ArtifactError(f"unsupported file type: {path} (expected .json, .jsonl or .csv)")
    try:
        return _validate_content(args, path, suffix)
    except OSError as exc:
        raise ArtifactError(f"cannot read file: {path}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        return _invalid(args, path, str(exc))


def _validate_content(args: argparse.Namespace, path: Path, suffix: str) -> int:
    if suffix == ".csv":
        cases = load_cases(path)
        _emit(
            {"file": str(path), "kind": "dataset", "valid": True, "cases": len(cases)},
            args.format,
        )
        return 0
    if suffix == ".jsonl":
        if _first_jsonl_kind(path) in (
            "langgraph_instrumentation.eval_dataset_header",
            None,
        ) or _jsonl_is_dataset(path):
            cases = load_cases(path)
            _emit(
                {
                    "file": str(path),
                    "kind": "dataset",
                    "valid": True,
                    "cases": len(cases),
                },
                args.format,
            )
            return 0
        load_artifacts(path)
        _emit({"file": str(path), "kind": "artifacts", "valid": True}, args.format)
        return 0
    kind = json.loads(path.read_text(encoding="utf-8")).get("kind")
    if kind == "langgraph_instrumentation.eval_dataset":
        cases = load_cases(path)
        _emit(
            {"file": str(path), "kind": "dataset", "valid": True, "cases": len(cases)},
            args.format,
        )
        return 0
    loaded = load_any(path)
    _emit({"file": str(path), "kind": loaded.to_dict()["kind"], "valid": True}, args.format)
    return 0


def _first_jsonl_kind(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        return record.get("kind") if isinstance(record, dict) else None
    raise ValueError(f"artifact file is empty: {path}")


def _jsonl_is_dataset(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        return isinstance(record, dict) and "case_id" in record
    return False


def _report(args: argparse.Namespace) -> int:
    run = _require_run(args.run)
    comparison = load_comparison(args.comparison) if args.comparison is not None else None
    document = render_report(run, comparison=comparison, title=args.title)
    args.out.write_text(document, encoding="utf-8")
    _emit(
        {"report": str(args.out), "run_id": run.run_id, "passed": run.passed},
        args.format,
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from .inspect import summarize_failures, turn_evidence

    run = _require_run(args.run)
    if args.turn is not None and args.case is None:
        raise ArtifactError("--turn requires --case")
    if args.traces is not None and args.case is None:
        raise ArtifactError("--traces requires --case")
    if args.case is not None:
        evidence = turn_evidence(
            run,
            args.case,
            args.turn if args.turn is not None else 0,
            traces=_load_trace_bundles(args, run, args.case, args.turn or 0),
            max_text=args.max_text,
            max_calls=args.max_calls,
        )
        _emit(evidence.to_dict(), args.format)
        return 0
    try:
        summary = summarize_failures(run, max_cases=args.max_cases)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc
    _emit(summary.to_dict(), args.format)
    return 0


def _load_trace_bundles(
    args: argparse.Namespace, run: EvalRun, case_id: str, turn_index: int
) -> dict[str, TraceBundle] | None:
    if args.traces is None:
        return None
    from ..identifiers import TraceId
    from ..stores import SQLiteTraceStore
    from .inspect import turn_evidence

    if not args.traces.is_file():
        raise ArtifactError(f"trace database does not exist: {args.traces}")
    preliminary = turn_evidence(run, case_id, turn_index)
    bundles: dict[str, TraceBundle] = {}
    store = SQLiteTraceStore(args.traces, read_only=True)
    try:
        for trace_id in preliminary.trace_ids:
            try:
                bundle = store.get_trace(TraceId.from_hex(trace_id))
            except (TypeError, ValueError) as exc:
                raise ArtifactError(f"invalid trace reference {trace_id!r}: {exc}") from exc
            if bundle is not None:
                bundles[trace_id] = bundle
    finally:
        store.close()
    return bundles


def _propose(args: argparse.Namespace) -> int:
    from .proposals import EvidenceRef, propose

    base = _require_cases(args.base_dataset)
    candidates = _require_cases(args.candidate_cases)
    evidence: list[EvidenceRef] = []
    if args.evidence_run is not None:
        baseline = _require_run(args.evidence_run)
        evidence.append(
            EvidenceRef(kind=EvidenceKind.RUN, ref=baseline.run_id, note=str(args.evidence_run))
        )
    for trace_id in args.evidence_trace:
        if not str(trace_id).strip():
            raise ArtifactError("evidence trace IDs must not be empty")
        evidence.append(EvidenceRef(kind=EvidenceKind.TRACE, ref=str(trace_id)))
    try:
        proposal = propose(
            candidates,
            proposal_id=args.proposal_id,
            author=args.author,
            rationale=args.rationale,
            evidence=evidence,
            validate_against=base,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot create proposal: {exc}") from exc
    save_proposal(proposal, args.out)
    _emit(
        {
            "proposal": str(args.out),
            "proposal_id": proposal.proposal_id,
            "status": proposal.status.value,
            "validation_passed": proposal.validation.passed if proposal.validation else None,
            "findings": list(proposal.validation.findings) if proposal.validation else [],
        },
        args.format,
    )
    return 0 if (proposal.validation and proposal.validation.passed) else 1


def _proposal_validate(args: argparse.Namespace) -> int:
    from .proposals import validate_proposal

    proposal = _require_proposal(args.proposal)
    base = _require_cases(args.base_dataset) if args.base_dataset is not None else ()
    validation = validate_proposal(proposal, base)
    _emit(
        {
            "proposal": str(args.proposal),
            "proposal_id": proposal.proposal_id,
            "status": proposal.status.value,
            "validation_passed": validation.passed,
            "findings": list(validation.findings),
        },
        args.format,
    )
    return 0 if validation.passed else 1


def _proposal_review(args: argparse.Namespace) -> int:
    from .proposals import ProposalStatus, review_proposal

    proposal = _require_proposal(args.proposal)
    decision = ProposalStatus.ACCEPTED if args.decision == "accept" else ProposalStatus.REJECTED
    try:
        reviewed = review_proposal(
            proposal, reviewer=args.reviewer, decision=decision, note=args.note
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot review proposal: {exc}") from exc
    save_proposal(reviewed, args.out)
    _emit(
        {
            "proposal": str(args.out),
            "proposal_id": reviewed.proposal_id,
            "status": reviewed.status.value,
            "reviewer": reviewed.review.reviewer if reviewed.review else None,
        },
        args.format,
    )
    return 0


def _apply_proposal(args: argparse.Namespace) -> int:
    from .proposals import apply_proposal

    base = _require_cases(args.dataset)
    proposal = _require_proposal(args.proposal)
    try:
        merged = apply_proposal(base, proposal, replace=args.replace)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot apply proposal: {exc}") from exc
    save_cases(merged, args.out)
    _emit(
        {
            "dataset": str(args.out),
            "cases": len(merged),
            "proposal_id": proposal.proposal_id,
            "replaced": args.replace,
        },
        args.format,
    )
    return 0


def _invalid(args: argparse.Namespace, path: Path, message: str) -> int:
    _emit({"file": str(path), "valid": False, "error": message}, args.format)
    return 1


def _require_cases(path: Path) -> tuple[EvalCase, ...]:
    try:
        return load_cases(path)
    except (TypeError, ValueError, OSError) as exc:
        raise ArtifactError(f"cannot load dataset {path}: {exc}") from exc


def _require_run(path: Path) -> EvalRun:
    try:
        return load_run(path)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot load run {path}: {exc}") from exc


def _require_proposal(path: Path) -> EvalProposal:
    try:
        proposal = load_proposal(path)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot load proposal {path}: {exc}") from exc
    if not isinstance(proposal, EvalProposal):
        raise ArtifactError(f"not an evaluation proposal: {path}")
    return proposal


def _parse_rule(spec: str) -> MetricRule:
    parts = spec.split(":")
    if len(parts) not in (2, 3, 4):
        raise ArtifactError(f"invalid --rule {spec!r}; expected METRIC:DIRECTION[:ABS][:REL]")
    metric, direction = parts[0], parts[1]
    try:
        parsed_direction = MetricDirection(direction)
    except ValueError:
        raise ArtifactError(
            f"invalid direction in --rule {spec!r}; expected lower_is_better or higher_is_better"
        ) from None
    absolute, relative = 0.0, 0.0
    try:
        if len(parts) >= 3:
            absolute = float(parts[2])
        if len(parts) == 4:
            relative = float(parts[3])
    except ValueError:
        raise ArtifactError(f"invalid tolerance in --rule {spec!r}") from None
    if not metric.strip() or absolute < 0 or relative < 0:
        raise ArtifactError(f"invalid --rule {spec!r}")
    try:
        return MetricRule(
            metric, parsed_direction, absolute_tolerance=absolute, relative_tolerance=relative
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid --rule {spec!r}: {exc}") from exc


def _case_document(case: EvalCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "name": case.name,
        "turns": [
            {
                "input": turn.payload(),
                "checks": [
                    str(getattr(check, "name", type(check).__name__)) for check in turn.checks
                ],
            }
            for turn in case.turns
        ],
        "checks": [str(getattr(check, "name", type(check).__name__)) for check in case.checks],
    }


def _import_agent(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ArtifactError(f"invalid --agent {spec!r}; expected MODULE:ATTR")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ArtifactError(f"cannot import agent module {module_name!r}: {exc}") from exc
    try:
        resolved = getattr(module, attribute)
    except AttributeError:
        raise ArtifactError(f"agent module has no attribute {attribute!r}") from None
    if hasattr(resolved, "invoke") or hasattr(resolved, "ainvoke"):
        return resolved
    if callable(resolved):
        try:
            return resolved()
        except Exception as exc:
            raise ArtifactError(f"agent factory {spec!r} failed: {exc}") from exc
    raise ArtifactError(f"agent entrypoint {spec!r} is not an app or factory")


__all__ = ["ArtifactError", "main"]
