"""Static, self-contained HTML reports for evaluation runs and comparisons.

The output is one HTML file with inline CSS and no external references, so it
opens directly from disk with no server, network access or browser tooling. It
summarizes actionable findings; full trace payloads stay in the recording store
and are referenced by trace ID only.
"""

from __future__ import annotations

import html
import json

from ..models import JSONValue
from .aggregation import aggregate_run
from .comparison import ComparisonSummary
from .models import EvalRun

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;max-width:70rem;color:#111}
.pass{color:#0a6b2d;font-weight:bold}.fail{color:#a31414;font-weight:bold}
table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;font-size:.9rem}
th{background:#f2f2f2}code{font-size:.85rem}.muted{color:#555}
details{margin:.5rem 0}summary{cursor:pointer;font-weight:bold}
"""


def render_report(
    run: EvalRun,
    *,
    comparison: ComparisonSummary | None = None,
    title: str = "Evaluation report",
) -> str:
    """Render a run, with an optional baseline comparison, as static HTML."""
    if not isinstance(run, EvalRun):
        raise TypeError("run must be an EvalRun")
    if comparison is not None and not isinstance(comparison, ComparisonSummary):
        raise TypeError("comparison must be a ComparisonSummary or null")
    aggregate = aggregate_run(run)
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        f"<title>{_escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_escape(title)}</h1>",
        _run_summary(run, aggregate.passed),
        _failure_section(run),
        _cases_section(run),
        _metrics_section(run),
    ]
    if comparison is not None:
        parts.append(_comparison_section(comparison))
    parts.append(
        '<p class="muted">Full trace payloads are not embedded. '
        "Open them with the trace CLI or a Perfetto export using the trace IDs above.</p>"
        "</body></html>"
    )
    return "\n".join(parts)


def _run_summary(run: EvalRun, passed: bool) -> str:
    status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
    trials = len(run.trials)
    cases = len({trial.case_id for trial in run.trials})
    return (
        f"<p>Run <code>{_escape(run.run_id)}</code>: {status} "
        f"({cases} case(s), {trials} trial(s))</p>"
    )


def _failure_section(run: EvalRun) -> str:
    failed = [trial for trial in run.trials if not trial.passed]
    if not failed:
        return "<h2>Failures</h2><p>None.</p>"
    items = []
    for trial in failed:
        reasons = _trial_reasons(trial)
        items.append(
            f"<li><code>{_escape(trial.case_id)}</code> "
            f"(trial {trial.trial_index}, {_escape(trial.execution_status.value)}): "
            f"{_escape('; '.join(reasons) or 'no assessments passed')}" + "</li>"
        )
    return "<h2>Failures</h2><ul>" + "".join(items) + "</ul>"


def _cases_section(run: EvalRun) -> str:
    rows = []
    for trial in sorted(run.trials, key=lambda item: (item.case_id, item.trial_index)):
        badge = (
            '<span class="pass">pass</span>' if trial.passed else '<span class="fail">fail</span>'
        )
        traces = ", ".join(f"<code>{_escape(trace_id)}</code>" for trace_id in trial.trace_ids)
        rows.append(
            f"<tr><td>{_escape(trial.case_id)}</td><td>{trial.trial_index}</td>"
            f"<td>{badge}</td><td>{_escape(trial.execution_status.value)}</td>"
            f"<td>{traces or '<span class="muted">none</span>'}</td></tr>"
        )
    return (
        "<h2>Cases</h2><table><tr><th>Case</th><th>Trial</th><th>Result</th>"
        "<th>Execution</th><th>Trace references</th></tr>" + "".join(rows) + "</table>"
    )


def _metrics_section(run: EvalRun) -> str:
    aggregate = aggregate_run(run)
    rows = []
    for case in aggregate.cases:
        for name, metric in sorted(case.metrics.values.items()):
            rows.append(
                f"<tr><td>{_escape(case.case_id)}</td><td>{_escape(name)}</td>"
                f"<td>{_escape(str(metric.value))}</td>"
                f"<td>{_escape(metric.unit)}</td>"
                f"<td>{_escape(metric.source)}</td>"
                f"<td>{_escape(metric.scope.value)}</td>"
                f"<td>{_escape(metric.completeness.value)}</td>"
                f"<td>{_escape(metric.detail or '')}</td></tr>"
            )
    if not rows:
        return "<h2>Metrics</h2><p>No metrics recorded.</p>"
    return (
        "<h2>Metrics</h2><table><tr><th>Case</th><th>Metric</th><th>Value</th>"
        "<th>Unit</th><th>Source</th><th>Scope</th><th>Completeness</th>"
        "<th>Detail</th></tr>" + "".join(rows) + "</table>"
    )


def _comparison_section(comparison: ComparisonSummary) -> str:
    status = (
        '<span class="pass">PASS</span>' if comparison.passed else '<span class="fail">FAIL</span>'
    )
    parts = [
        f"<h2>Baseline comparison</h2><p>Baseline "
        f"<code>{_escape(comparison.baseline_run_id)}</code> vs candidate "
        f"<code>{_escape(comparison.candidate_run_id)}</code>: {status}</p>",
        _id_list("Regressed", comparison.regressed_case_ids, "fail"),
        _id_list("Improved", comparison.improved_case_ids, "pass"),
        _id_list("Unchanged", comparison.unchanged_case_ids, ""),
    ]
    for mismatch in comparison.identity_mismatches:
        parts.append(
            f"<p>Identity mismatch <code>{_escape(mismatch.key)}</code>: "
            f"baseline <code>{_escape(_json(mismatch.baseline))}</code> vs "
            f"candidate <code>{_escape(_json(mismatch.candidate))}</code></p>"
        )
    details = []
    for case in comparison.cases:
        for metric in case.metrics:
            details.append(
                f"<tr><td>{_escape(case.case_id)}</td>"
                f"<td>{_escape(metric.metric)}</td>"
                f"<td>{_escape(metric.kind.value)}</td>"
                f"<td>{_escape(str(metric.delta))}</td>"
                f"<td>{_escape(metric.reason or '')}</td></tr>"
            )
    if details:
        parts.append(
            "<details><summary>Metric comparisons</summary><table>"
            "<tr><th>Case</th><th>Metric</th><th>Outcome</th><th>Delta</th>"
            "<th>Reason</th></tr>" + "".join(details) + "</table></details>"
        )
    for reason in comparison.incompatible_reasons:
        parts.append(f"<p>Incompatible: {_escape(reason)}</p>")
    return "\n".join(parts)


def _id_list(label: str, case_ids: tuple[str, ...], css: str) -> str:
    if not case_ids:
        return f"<p>{label}: none.</p>"
    items = "".join(f"<li><code>{_escape(case_id)}</code></li>" for case_id in case_ids)
    cls = f' class="{css}"' if css else ""
    return f"<p{cls}>{label}:</p><ul>{items}</ul>"


def _trial_reasons(trial: object) -> list[str]:
    reasons: list[str] = []
    assessments = getattr(trial, "assessments", ())
    turns = getattr(trial, "turns", ())
    for assessment in (*assessments, *(a for turn in turns for a in turn.assessments)):
        if assessment.status.value in ("fail", "error") and assessment.message:
            reasons.append(f"{assessment.evaluator}: {assessment.message}")
    error = getattr(trial, "error", None)
    if error is not None:
        reasons.append(f"execution: {error.exception_type}")
    return reasons


def _json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True)


def _escape(value: str) -> str:
    return html.escape(value, quote=True)
