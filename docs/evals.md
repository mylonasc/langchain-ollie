# Evaluate agent behavior and performance

Use evaluations to protect both answer quality and execution efficiency. A case
describes a conversation and its checks; the runner executes it under tracing,
then records answers, measurements and assessments. You can repeat cases to
detect inconsistent behavior and compare a candidate agent with a baseline.

This guide starts with a local example, then covers saved results and automation.
Saving, inspecting and comparing results require no service or database. Running
an agent uses whatever tools and providers that application configures; the
examples here use scripted models and local tools, so they run offline.

## Run your first evaluation

Start with a small behavior you can verify independently. The repository's
addition agent is useful because you can check the answer, require a real tool
call, and cap the number of model calls.

From a checkout, install the example dependencies:

```bash
uv sync --group dev --group agents
uv run --no-sync python examples/local_eval.py
```

The [example](../examples/local_eval.py) prints `Case add-two-numbers: PASS`, followed
by elapsed time, token usage, model/tool call counts and trace IDs. Its checks are:

```python
from langchain_ollie import EvalCase, EvalTurn, ExactText, MaxMetric, ToolCalled

case = EvalCase(
    case_id="add-two-numbers",
    turns=(
        EvalTurn.user(
            "Add 2 and 3",
            checks=(ExactText("Result: 5"), ToolCalled("add")),
        ),
    ),
    checks=(MaxMetric("llm.call_count", 2),),
)
```

Turn checks test the answer and evidence for that turn. Case checks apply to the
whole conversation, so the two-call budget also works when a case has several
turns. For your own application, run the case with
`EvalRunner(LangGraphEvalAdapter(app)).run([case])`; use `await runner.arun(...)`
when invoking an async application. The runner creates a fresh recording session
for each case trial. Keep any custom session/store factories equally isolated.

`LangGraphEvalAdapter` accumulates message history by default. If the application
owns a checkpointer, choose `state_mode="checkpointer"` to use a trial-specific
thread ID instead. See the [evaluation classes](uml.md#evaluation-classes) for how
the runner, adapter and checks collaborate.

Session and conversation isolation do not reset arbitrary model or tool state.
Ensure your application can serve repeated invocations before running several
cases or trials. In particular, the short scripted addition model is single-use;
the [workflow verification example](agent-eval-workflow.md#verify-the-expanded-dataset)
builds a fresh agent for each case.

## Interpret results and missing evidence

A failed answer, an execution error and an unavailable measurement need different
responses. Read execution status before interpreting check results: an exception
is not simply an incorrect answer, and later turns may be skipped.

Every metric carries a unit, source, scope and completeness. `tokens.*` reflects
reported model usage; absent usage is **missing**, not zero. Scripted examples
provide deterministic usage values, not a measurement of a real provider's cost.
CPU/RSS metrics describe the entire process, not the individual agent.

`MaxMetric` and `MinMetric` let you choose how a missing metric affects a check:
fail, error or skip. Inspect completeness as well as value when traces are partial.
For visualization limits, see [interpreting traces](../README.md#view-locally).

## Save and reopen results

Save a run when you need reproducible comparisons, reports or evidence for a
regression case. Given a completed `run`:

```python
from langchain_ollie import load_run, save_run

save_run(run, "run.json")
saved_run = load_run("run.json")
assert saved_run.passed == run.passed
```

Datasets and results serve different purposes. Use `save_cases` / `load_cases`
for reusable test definitions, and the following functions for observed results:

| Result | Saved by | Reopened with |
| --- | --- | --- |
| `EvalRun` | `save_run` / CLI `run` | `load_run` |
| `RunAggregate` | `save_aggregate` | `load_aggregate` |
| `ComparisonSummary` | `save_comparison` / CLI `compare --out` | `load_comparison` |

Files are versioned so a result is not silently interpreted using a different
schema:

- Single artifact files end with `.json`. Canonical serialization makes saved
  results stable to review and compare.
- `.jsonl` files hold one artifact per line (runs, aggregates and comparisons
  may mix). Use `save_artifacts` / `load_artifacts`, or `load_any` for a single
  `.json` file of any kind.
- Artifacts carry trace references (`trace_ids`), never full trace payloads.
  Retain traces separately if you need detailed evidence later. The default
  in-memory recording does not create a persistent trace database; configure
  dedicated stores through the Python runner's `store_factory` for durable trials.
  Use a fresh database per trial and close caller-owned stores yourself.
  See [SQLite recording and inspection](../README.md#durable-local-recording).
- Unknown `kind` values and unsupported `schema_version` values fail closed with
  the file path in the error. Custom evaluator callables (`FunctionEvaluator`,
  `JudgeEvaluator`) are never serialized; `StatePredicate` and the deterministic
  checks are. Reattach custom checks programmatically after `load_cases`.
- Datasets are separate artifacts, saved with `save_cases` and reopened with
  `load_cases`; they follow the same versioning discipline.

Run artifacts can still contain agent answers, inputs and check evidence.
Metadata-only tracing does not make those fields safe to publish; follow the
[security guidance](../SECURITY.md#protecting-recordings).

## Compare a candidate with a baseline

Use the same cases and compatible run identities for the baseline and candidate.
Comparison aggregates repeated trials first. The default worst-trial policy is
conservative: an intermittent failure should not disappear behind a successful
retry. Choose other trial or metric policies deliberately.

Quality regressions always matter. Performance metrics gate only when you name
them in a rule. For example:

```bash
python -m langchain_ollie evals compare base.json cand.json \
  --rule "tokens.total:lower_is_better:10:0.05" --out comparison.json
```

This flags a token increase beyond the allowed tolerance: the larger of 10 tokens
or 5% of the baseline. Repeat `--rule` for other metrics, using
`higher_is_better` when an increase is desirable. Missing-metric behavior can be
configured with `MetricRule` in the Python API.

Keep `dataset.id`, `agent.id`, `eval.config` and `model.id` metadata compatible
where present; mismatched identities are rejected rather than compared silently.
Added or removed cases are incompatible by default; `--allow-added-removed`
explicitly permits them. To turn a finding into a regression case, follow the
[reviewed dataset workflow](agent-eval-workflow.md).

## Run evaluations from the command line

The CLI is useful for repeatable local runs and CI gates. First save cases with
`save_cases` and make your agent importable. `--agent MODULE:ATTR` accepts an
application with `invoke`/`ainvoke`, or a zero-argument factory returning one.
In a repository checkout, prefix commands with `uv run --no-sync` to use the
installed project environment.

```bash
python -m langchain_ollie evals run cases.json \
  --agent mymodule:build_app --out run.json --trials 3
python -m langchain_ollie evals cases cases.json
python -m langchain_ollie evals show run.json
python -m langchain_ollie evals compare base.json cand.json \
  --rule "tokens.total:lower_is_better:10:0.05" --out comparison.json
python -m langchain_ollie evals validate cases.json
python -m langchain_ollie evals report run.json \
  --comparison comparison.json --out report.html
```

The example names `cases.json` and `mymodule:build_app` stand for your dataset and
application factory. `--trials 3` runs each case three times, independently.
`--capture` defaults to `metadata`; choose `bounded` or `full` only when the extra
trace content is necessary and safe to retain.

### Inspect failures and write reports

Start with a bounded summary, then inspect the relevant case and turn. This keeps
large runs manageable and makes absent evidence explicit.

```bash
python -m langchain_ollie evals inspect run.json
python -m langchain_ollie evals inspect run.json --case add --turn 0
```

If you retained a trace database for that trial, add `--traces trial.db` to resolve
tool/model calls through a read-only connection. Without bundles, trace IDs remain
references; the inspector does not reconstruct missing calls. Respect truncation
and missing-evidence markers when writing findings.

Use `report` to share a browsable summary of failures, measurements and comparison
outcomes. The generated HTML report opens locally without a service. For proposed
dataset changes, use the [proposal review walkthrough](agent-eval-workflow.md#propose-a-stronger-case).

## Output and exit codes

Use exit codes to distinguish a failed evaluation from a command that could not
run. Commands print JSON by default for automation; add `--format text` for a
concise terminal summary.

| Code | Meaning |
| --- | --- |
| 0 | Success; for `run` the run passed, for `compare` no regression. |
| 1 | Gate failed: the run failed, the comparison regressed (or is incompatible), or validation found a problem. |
| 2 | Usage or artifact error: bad arguments, missing files, unreadable or wrong-kind artifacts. |

## Compatibility policy

Keep original datasets and runs when upgrading so you can reproduce earlier
decisions. Artifact `kind` markers are stable; unsupported schema versions are
rejected with an explicit error instead of silently reinterpreted. CLI flags are
additive and exit-code meanings remain stable. See the
[version policy](../CHANGELOG.md#version-policy) for the broader API guarantees.
