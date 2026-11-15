# Evaluation artifacts and CLI

Local-first persistence, inspection and regression gates for the evaluation
framework. Everything here is offline: JSON files on disk, a static HTML report
and process exit codes. No server, database or network access.

For how execution adapters, recording sessions and evaluators work together, see
the [evaluation class diagram](uml.md#evaluation-classes) and
[architecture overview](architecture.md#evaluations-build-on-tracing).

## Artifacts

Three versioned kinds, each with a `kind` marker and a `schema_version`:

| Kind | Marker | Saved by |
| --- | --- | --- |
| `EvalRun` | `langgraph_instrumentation.eval_run` | `save_run` / CLI `run` |
| `RunAggregate` | `langgraph_instrumentation.eval_aggregate` | `save_aggregate` |
| `ComparisonSummary` | `langgraph_instrumentation.eval_comparison` | `save_comparison` / CLI `compare --out` |

Rules:

- Single artifact files end with `.json` and contain one canonical JSON document
  (`sort_keys=True`, `indent=2`, trailing newline). Saving a loaded artifact
  reproduces the source bytes exactly.
- `.jsonl` files hold one artifact per line (runs, aggregates and comparisons
  may mix). Use `save_artifacts` / `load_artifacts`, or `load_any` for a single
  `.json` file of any kind.
- Artifacts carry trace references (`trace_ids`), never trace payloads. Full
  spans and events stay in the recording store; open them with the trace CLI
  or a Perfetto export.
- Unknown `kind` values and unsupported `schema_version` values fail closed with
  the file path in the error. Custom evaluator callables (`FunctionEvaluator`,
  `JudgeEvaluator`) are never serialized; `StatePredicate` and the deterministic
  checks are. Reattach custom checks programmatically after `load_cases`.
- Datasets are separate (`langgraph_instrumentation.eval_dataset`, see
  `save_cases` / `load_cases`) and follow the same versioning discipline.

## CLI

All commands go through `python -m langgraph_instrumentation evals`. Every
command is a thin wrapper over the public Python API of the same name.

```bash
python -m langgraph_instrumentation evals run cases.json \
  --agent mymodule:build_app --out run.json --trials 3
python -m langgraph_instrumentation evals cases cases.json
python -m langgraph_instrumentation evals show run.json
python -m langgraph_instrumentation evals compare base.json cand.json \
  --rule "tokens.total:lower_is_better:10:0.05" --out comparison.json
python -m langgraph_instrumentation evals validate cases.json
python -m langgraph_instrumentation evals report run.json \
  --comparison comparison.json --out report.html
```

Notes:

- `--agent MODULE:ATTR` resolves an attribute that is either an app (has
  `invoke`/`ainvoke`) or a zero-argument factory returning one. The module must
  be importable (install the package or set `PYTHONPATH`). The adapter attaches
  `session.handler` through the runtime config exactly like the Python API.
- `--capture` defaults to `metadata`. `bounded`/`full` capture more payload;
  prefer the default for shared or committed artifacts.
- `--rule METRIC:DIRECTION[:ABS][:REL]` is repeatable. Direction is
  `lower_is_better` or `higher_is_better`; tolerances are non-negative numbers.
- `compare` also accepts `--trials-policy`, `--metrics-policy` and
  `--allow-added-removed`. Identity keys (`dataset.id`, `agent.id`,
  `eval.config`, `model.id`) default to the comparison defaults.
- `report` writes one self-contained HTML file: inline CSS, no external
  references, no scripts. It lists failures first, then cases, metrics with
  provenance, trace references and the comparison outcome.
- `inspect` summarizes failures and turn evidence without touching trace
  stores; `--traces DB` additionally resolves tool/LLM calls from a
  caller-supplied read-only trace database.
- `propose` builds a candidate proposal from a candidate-cases dataset plus
  rationale, author and evidence references. `proposal-validate` re-checks it,
  `proposal-review` records an explicit accept/reject decision, and
  `apply-proposal` merges an accepted proposal into a golden dataset
  (`--replace` to revise colliding cases).

## Output and exit codes

Commands print machine-readable JSON to stdout by default (`--format text` for
a concise human rendering; JSON keys are sorted for stability).

| Code | Meaning |
| --- | --- |
| 0 | Success; for `run` the run passed, for `compare` no regression. |
| 1 | Gate failed: the run failed, the comparison regressed (or is incompatible), or validation found a problem. |
| 2 | Usage or artifact error: bad arguments, missing files, unreadable or wrong-kind artifacts. |

## Compatibility policy

Artifact `kind` markers are stable. A `schema_version` bump means older
artifacts fail closed with an explicit error rather than being silently
reinterpreted. CLI flags are additive; exit-code semantics above are stable.
