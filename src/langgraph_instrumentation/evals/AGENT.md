# Evaluation Guide

## Scope

This package converts recorded executions into reproducible cases, trace-derived
measurements, assessments and portable run artifacts. It depends on neutral models,
the `RecordingSession` facade, capture policies and evaluator protocols. It must not
import concrete stores, exporters, provider SDKs or an agent runtime into its core
models.

## Contracts

- Definitions, observed execution, measurements, judgments and comparisons are
  separate records. Execution failures never become ordinary failed assessments.
- Run every case trial in a fresh, empty recording session. Never let one case
  observe another case's traces or conversation state.
- Reuse the recorder and neutral records. Do not build a parallel timeline or
  duplicate full trace payloads in evaluation artifacts; retain trace references.
- Every metric states its source, unit, scope and completeness. Missing evidence is
  `missing`, never zero. Process CPU/RSS remains process-wide.
- Capture policy still governs instrumentation payloads. Evaluation storage of
  outputs is explicit and bounded; a runner must not silently enable full tracing.
- Execution adapters attach `session.handler` to the application's runtime config
  and normalize outputs without importing framework details into neutral records.
- Dataset and run artifacts carry a schema version. Unsupported versions fail
  closed; custom evaluator callables are attached programmatically, not serialized.
  `StatePredicate` is portable; `FunctionEvaluator` and `JudgeEvaluator` are not.
- Comparisons are deterministic artifacts: aggregate trials first, then compare
  baseline and candidate runs case by case. Quality regression always regresses;
  metrics only gate when a `MetricRule` names them. Identity keys must agree or
  the comparison fails instead of comparing silently.
- Persisted artifacts reference traces by ID and never embed trace payloads.
  Canonical JSON (sorted keys) round-trips byte-stable; JSONL holds one artifact
  per line with kind dispatch.
- The CLI is a thin wrapper over the public Python API: agent entrypoints are
  `MODULE:ATTR` app-or-factory references, trace bundles resolve from a
  caller-supplied read-only database (`inspect --traces`), output is stable JSON
  by default, and exit codes are 0 success/pass, 1 gate failed, 2 usage/artifact
  error. Reports are static, self-contained HTML with no external references.
- Agent-facing inspection consumes run artifacts and caller-supplied bundles
  only: no store imports, no direct database queries. Summaries are versioned,
  bounded and flag truncation; missing trace bundles are reported, never
  synthesized.
- Proposals are immutable candidate records. Validation is structural and never
  executes an agent. Review requires explicit reviewer and decision arguments;
  the input is never mutated. Only accepted proposals apply, and golden
  collisions fail without explicit `--replace`. Feedback content is
  caller-supplied and bounded; instrumentation payloads are never captured
  implicitly.
- The default suite is offline. Fake models, not agent runtimes; optional judges and
  provider calls require deliberate configuration.

## Tests

`tests/test_evals.py` covers real LangGraph execution, multi-turn state isolation,
metrics provenance, failure/skip/timeout/interrupt/incomplete semantics, sampler
reachability, dataset round-trips and async execution.
`tests/test_eval_comparison.py` covers fail-safe trial aggregation, metric modes,
tolerances, missing-metric policies, identity mismatches, added/removed cases and
optional judges.
`tests/test_eval_artifacts.py` covers canonical persistence, byte-stability,
kind/version dispatch, JSONL, proposal/feedback artifacts and no-payload-embedding.
`tests/test_eval_cli.py` covers the thin CLI surface, gate exit codes, the
offline HTML report, proposal lifecycle commands and read-only trace resolution.
`tests/test_eval_inspect.py` covers bounded failure/turn/regression summaries,
missing-trace reporting and truncation flags.
`tests/test_eval_proposals.py` covers feedback validation, proposal validation,
explicit review, golden-dataset application and artifact persistence.
`tests/test_eval_demo.py` pins the release gate: baseline fails golden budgets,
optimized passes, reversed comparison regresses. `examples/` holds runnable
demos with committed versioned datasets; demo scripts exit non-zero on
violated expectations and write artifacts outside the repo.
