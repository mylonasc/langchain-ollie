# Agent evaluation workflow

Instructions for coding and workflow agents that diagnose evaluation failures
and propose targeted evaluation improvements. Everything here uses only public
Python and CLI interfaces (`langgraph_instrumentation` top-level exports and
`python -m langgraph_instrumentation evals …`). No internal modules, no direct
store or database access, no MCP server, no network calls. All commands run
offline from the repository root.

## The loop

1. **Run** a dataset against an agent and keep the artifact.
2. **Inspect** the failure: which turn, which tool or LLM call, which metric.
3. **Compare** against the baseline to see what regressed.
4. **Propose** a candidate evaluation case from the evidence.
5. **Validate** the proposal, get it **reviewed**, and **apply** the accepted
   case to a golden dataset.
6. **Re-run** and compare again to confirm the outcome.

Read access is the default. Creating or accepting proposals is explicit and
always carries author, rationale and reviewer provenance.

## Reproducible walkthrough

The scripted agent below answers `Result: 5` to `Add 2 and 3` and needs the
`add` tool, so every step runs without credentials or services.

```bash
# 0. A one-turn golden dataset (JSON, JSONL and CSV all work).
python - <<'EOF'
from langgraph_instrumentation import EvalCase, EvalTurn, ExactText, save_cases
save_cases(
    [EvalCase(case_id="add",
              turns=(EvalTurn.user("Add 2 and 3", checks=(ExactText("Result: 5"),)),))],
    "/tmp/opencode/walk-cases.json",
)
EOF

# 1. Run it. Exit code 0 means the run passed; 1 means it failed.
PYTHONPATH=examples python -m langgraph_instrumentation evals run \
  /tmp/opencode/walk-cases.json --agent local_agent:build_agent \
  --out /tmp/opencode/walk-run.json

# 2. Inspect: failure summary, then evidence for the turn that failed.
python -m langgraph_instrumentation evals inspect /tmp/opencode/walk-run.json
python -m langgraph_instrumentation evals inspect /tmp/opencode/walk-run.json \
  --case add --turn 0

# 3. Compare a follow-up run against this baseline. Exit code 1 on regression.
PYTHONPATH=examples python -m langgraph_instrumentation evals run \
  /tmp/opencode/walk-cases.json --agent local_agent:build_agent \
  --out /tmp/opencode/walk-cand.json
python -m langgraph_instrumentation evals compare \
  /tmp/opencode/walk-run.json /tmp/opencode/walk-cand.json \
  --rule "tokens.total:lower_is_better:10" --out /tmp/opencode/walk-cmp.json

# 4. Draft a candidate case file for the newly observed behavior and propose it.
python - <<'EOF'
from langgraph_instrumentation import EvalCase, EvalTurn, ExactText, save_cases
save_cases(
    [EvalCase(case_id="add-then-explain",
              turns=(EvalTurn.user("Add 2 and 3, then explain the steps",
                                  checks=(ExactText("Result: 5"),)),))],
    "/tmp/opencode/walk-candidate.json",
)
EOF
python -m langgraph_instrumentation evals propose \
  --base-dataset /tmp/opencode/walk-cases.json \
  --candidate-cases /tmp/opencode/walk-candidate.json \
  --rationale "Cover multi-step explanations observed in recent runs." \
  --author agent --evidence-run /tmp/opencode/walk-run.json \
  --out /tmp/opencode/walk-proposal.json
# Exit code 1 here means validation findings: read them, do not ignore them.

# 5. Validate explicitly, then record an accept/reject review.
python -m langgraph_instrumentation evals proposal-validate \
  /tmp/opencode/walk-proposal.json --base-dataset /tmp/opencode/walk-cases.json
python -m langgraph_instrumentation evals proposal-review \
  /tmp/opencode/walk-proposal.json --decision accept --reviewer sme \
  --note "Reproduces the observed gap." --out /tmp/opencode/walk-accepted.json

# 6. Merge the accepted case into a golden dataset and re-run the loop.
python -m langgraph_instrumentation evals apply-proposal \
  /tmp/opencode/walk-cases.json /tmp/opencode/walk-accepted.json \
  --out /tmp/opencode/walk-golden.json
python -m langgraph_instrumentation evals cases /tmp/opencode/walk-golden.json
```

## Rules for agents

- Read runs with `inspect` and `show`; compare runs with `compare`. Never open
  trace stores directly — request trace bundles through the caller or work from
  the IDs in the evidence. IDs without bundles are reported as missing.
- Keep every number with its unit, source, scope and completeness. Process
  CPU/RSS is process-wide; never present it as per-agent attribution.
- A proposal must carry rationale, evidence references and fully specified
  candidate cases. Custom code evaluators (`FunctionEvaluator`,
  `JudgeEvaluator`) cannot be serialized — propose portable checks only.
- A candidate is not a golden case. Only `proposal-review` followed by
  `apply-proposal` makes it golden, and collisions without `--replace` fail
  instead of silently overwriting.
- Bounded output is a contract: summaries truncate and say so. If evidence is
  marked truncated or missing, say so in findings instead of guessing.
- MCP or other tool adapters are intentionally not part of this workflow; the
  CLI surface above already exposes every bounded operation with the same
  authorization model (read by default, explicit flags for writes).
