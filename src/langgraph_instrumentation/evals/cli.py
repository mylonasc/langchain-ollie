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

from ..policies import CapturePolicy
from .aggregation import MetricAggregation, TrialAggregation, aggregate_run
from .artifacts import (
    load_any,
    load_artifacts,
    load_comparison,
    load_run,
    save_comparison,
    save_run,
)
from .comparison import MetricDirection, MetricRule, compare_runs
from .datasets import load_cases
from .models import EvalCase, EvalRun
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
