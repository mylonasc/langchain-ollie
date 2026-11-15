# Improve an evaluation dataset from evidence

A useful regression case protects behavior you care about, not just the answer
from one successful run. This workflow helps a developer or coding agent inspect
results, propose a focused check, and add it to the golden dataset only after
review. A **golden dataset** is the reviewed set of cases you use to judge future
changes.

Start with the [evaluation guide](evals.md) if you have not run a case yet. The
walkthrough below uses the repository's scripted addition agent and needs no
credentials or external services. For a complete optimization example with a
failing baseline, see the [end-to-end demonstration](../README.md#end-to-end-demonstration).

## The improvement loop

Keep diagnosis and dataset changes separate. Inspection is read-only; changing
the golden dataset requires an explicit proposal and review.

1. **Run** a dataset and retain the result.
2. **Inspect** the relevant turns, assessments and available trace evidence.
3. **Compare** a candidate run against the baseline.
4. **Propose** a case that protects the behavior you identified.
5. **Validate and review** the proposal before applying it.
6. **Re-run** the expanded dataset to confirm what it actually checks.

## Prepare a local baseline

Use a small passing baseline so that any later change is easy to explain. From
the repository root, install the example dependencies and create a workspace for
the generated files:

```bash
uv sync --group dev --group agents
mkdir -p /tmp/langgraph-eval-walkthrough
```

Create a one-turn dataset that checks the answer:

```bash
uv run --no-sync python - <<'EOF'
from langchain_ollie import EvalCase, EvalTurn, ExactText, save_cases

save_cases(
    [EvalCase(
        case_id="add",
        turns=(EvalTurn.user("Add 2 and 3", checks=(ExactText("Result: 5"),)),),
    )],
    "/tmp/langgraph-eval-walkthrough/cases.json",
)
EOF

PYTHONPATH=examples uv run --no-sync python -m langchain_ollie evals run \
  /tmp/langgraph-eval-walkthrough/cases.json --agent local_agent:build_agent \
  --out /tmp/langgraph-eval-walkthrough/base.json
```

The run should pass. `PYTHONPATH=examples` makes the repository's example factory
importable; replace `local_agent:build_agent` with your own factory for a real
application.

## Inspect before changing the test

A passing answer alone does not prove that the intended tool was used. Inspect
the saved result to see which claims the current checks support:

```bash
uv run --no-sync python -m langchain_ollie evals inspect \
  /tmp/langgraph-eval-walkthrough/base.json
uv run --no-sync python -m langchain_ollie evals inspect \
  /tmp/langgraph-eval-walkthrough/base.json --case add --turn 0
```

The failure summary is empty because this baseline passes. Turn evidence includes
the answer, metrics and trace references. This CLI run uses in-memory recording;
its saved run does not retain full trace bundles. Do not infer individual tool
calls from trace IDs alone. When durable traces are available, supply a trial's
database with `--traces trial.db`, as described in the
[inspection guide](evals.md#inspect-failures-and-write-reports).

To establish comparison behavior, run the unchanged agent again:

```bash
PYTHONPATH=examples uv run --no-sync python -m langchain_ollie evals run \
  /tmp/langgraph-eval-walkthrough/cases.json --agent local_agent:build_agent \
  --out /tmp/langgraph-eval-walkthrough/candidate.json
uv run --no-sync python -m langchain_ollie evals compare \
  /tmp/langgraph-eval-walkthrough/base.json \
  /tmp/langgraph-eval-walkthrough/candidate.json \
  --rule "tokens.total:lower_is_better:10" \
  --out /tmp/langgraph-eval-walkthrough/comparison.json
```

No regression is expected. In an actual investigation, replace the second run
with your candidate implementation and explain the measured difference. See
[comparison rules](evals.md#compare-a-candidate-with-a-baseline) for tolerances and
missing-metric behavior.

## Propose a stronger case

The baseline checks only the final text. Add a separate case requiring both the
correct answer and a recorded `add` call, so a future implementation cannot pass
by returning a hard-coded answer alone. This is a deliberate coverage improvement,
not a claim that the passing baseline exhibited a failure.

```bash
uv run --no-sync python - <<'EOF'
from langchain_ollie import EvalCase, EvalTurn, ExactText, ToolCalled, save_cases

save_cases(
    [EvalCase(
        case_id="add-with-tool",
        turns=(EvalTurn.user(
            "Use the add tool to add 2 and 3",
            checks=(ExactText("Result: 5"), ToolCalled("add")),
        ),),
    )],
    "/tmp/langgraph-eval-walkthrough/candidate-cases.json",
)
EOF

uv run --no-sync python -m langchain_ollie evals propose \
  --base-dataset /tmp/langgraph-eval-walkthrough/cases.json \
  --candidate-cases /tmp/langgraph-eval-walkthrough/candidate-cases.json \
  --rationale "The baseline checks text only; also protect use of the addition tool." \
  --author developer --evidence-run /tmp/langgraph-eval-walkthrough/base.json \
  --out /tmp/langgraph-eval-walkthrough/proposal.json
```

The proposal contains candidate cases, rationale and evidence references. It does
not change the golden dataset or execute the proposed cases.

## Validate, review and apply

Structural validation catches invalid cases and ID collisions. It cannot decide
whether a check reflects your intended behavior; that is the reviewer's job.
Read validation findings before recording a decision, and accept only after
reviewing the candidate checks and rationale.

```bash
uv run --no-sync python -m langchain_ollie evals proposal-validate \
  /tmp/langgraph-eval-walkthrough/proposal.json \
  --base-dataset /tmp/langgraph-eval-walkthrough/cases.json
uv run --no-sync python -m langchain_ollie evals proposal-review \
  /tmp/langgraph-eval-walkthrough/proposal.json --decision accept --reviewer maintainer \
  --note "Protects tool use as well as the final answer." \
  --out /tmp/langgraph-eval-walkthrough/accepted.json
uv run --no-sync python -m langchain_ollie evals apply-proposal \
  /tmp/langgraph-eval-walkthrough/cases.json /tmp/langgraph-eval-walkthrough/accepted.json \
  --out /tmp/langgraph-eval-walkthrough/golden.json
```

Use `--decision reject` when the proposal is not ready. Only accepted proposals
can be applied. Existing case IDs are not overwritten unless you explicitly use
`--replace`.

## Verify the expanded dataset

Applying a proposal is not proof that its checks work. List the merged cases and
execute them before relying on the new dataset. The addition example's scripted
model contains only one invocation's responses, so build a fresh agent for each
case rather than reusing an exhausted script:

```bash
uv run --no-sync python -m langchain_ollie evals cases \
  /tmp/langgraph-eval-walkthrough/golden.json
PYTHONPATH=examples uv run --no-sync python - <<'EOF'
from local_agent import build_agent
from langchain_ollie import EvalRunner, LangGraphEvalAdapter, load_cases, save_run

for case in load_cases("/tmp/langgraph-eval-walkthrough/golden.json"):
    run = EvalRunner(LangGraphEvalAdapter(build_agent())).run([case])
    save_run(run, f"/tmp/langgraph-eval-walkthrough/{case.case_id}-verified.json")
    print(f"{case.case_id}: {'PASS' if run.passed else 'FAIL'}")
    assert run.passed
EOF
```

Both cases should pass. Recording isolation does not reset arbitrary state held
by an application's model or tools. The scripted agent follows a fixed addition
sequence, so this walkthrough demonstrates the review process, not general
arithmetic reasoning. The [optimization demo](../README.md#end-to-end-demonstration)
also demonstrates a baseline that fails a newly added budget check.

## Guidance for automated reviewers

Automated findings should remain auditable and limited to available evidence.
Use the same public inspection and proposal interfaces as a human reviewer:

- Keep every measurement with its unit, source, scope and completeness. Process
  CPU/RSS is process-wide, never per-agent attribution.
- Treat missing trace bundles and truncated summaries as limits on a finding,
  not an invitation to reconstruct or guess the missing evidence.
- Include rationale, evidence references and fully specified candidate cases.
  Use portable checks in saved datasets; reattach `FunctionEvaluator` or
  `JudgeEvaluator` programmatically when needed.
- Do not treat an agent-generated proposal as accepted. Record an explicit
  reviewer and decision before applying it to the golden dataset.
- Protect saved outputs and evidence using the [security guidance](../SECURITY.md).
  Read the [exit-code reference](evals.md#output-and-exit-codes) to distinguish
  failed checks from invalid commands or unreadable artifacts.
