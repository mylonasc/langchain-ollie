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
- The default suite is offline. Fake models, not agent runtimes; optional judges and
  provider calls require deliberate configuration.

## Tests

`tests/test_evals.py` covers real LangGraph execution, multi-turn state isolation,
metrics provenance, failure/skip semantics, dataset round-trips and async execution.
`tests/test_eval_comparison.py` covers fail-safe trial aggregation, metric modes,
tolerances, missing-metric policies, identity mismatches, added/removed cases and
optional judges. Add contract tests at the model, adapter, metric or evaluator
boundary changed.
