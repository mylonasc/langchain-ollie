# Changelog

Use these notes to understand changes in behavior and plan upgrades. For a first
integration, start with the [quickstart](README.md#quickstart-trace-a-langgraph-agent);
for component responsibilities, see the [architecture overview](docs/architecture.md).

## 0.1.0 — local-first API (unreleased)

This release replaces the prototype with explicit recording sessions, local
artifacts and evaluations built on the same execution evidence.

- The distribution is now `langchain-ollie`. Import public APIs from
  `langchain_ollie` and run the CLI with `python -m langchain_ollie`. Update imports
  in existing applications; no alias package is installed. Saved evaluation formats and SQLite
  schemas are unchanged, so the rename does not require rewriting recorded data.

- Modern LangGraph/Deep Agents callbacks, bounded metadata-first recording,
  streaming summaries, explicit causal links and isolated destinations.
- Deterministic atomic Perfetto export and transactional SQLite v1 with read-only
  inspection/export CLI and explicit stopped-writer crash recovery.
- Real offline agent coverage on standard CPython 3.12–3.14.
- Optional process-wide CPU/RSS counters and SDK-backed export-only OTel bridge.
  The profiler compatibility spike ended in an evidenced no-go for automatic
  py-spy/Speedscope timeline correlation; see [ADR-001](docs/adr/001-profiler-correlation.md).
- Local evaluation foundation: immutable cases/trials, a sync/async execution
  adapter for LangGraph, per-trial recording isolation, trace-derived token,
  latency, tool and optional process metrics with explicit provenance, composable
  deterministic checks and time/resource budgets. JSON, JSONL and CSV datasets
  load without a provider.
- Evaluation comparisons: fail-safe repeated-trial aggregation, baseline
  comparison with explicit metric rules/tolerances/directions, structured state
  predicates and an optional provider-free judge protocol. Execution, assessment
  and missing-evidence outcomes stay distinct.
- Agent-facing inspection and proposal workflows: bounded read-only failure,
  turn and regression summaries over run artifacts plus caller-supplied trace
  bundles (missing evidence explicit, truncation flagged); feedback records
  with provenance and a structural privacy boundary; candidate proposals with
  rationale, evidence refs and fully specified cases; structural validation,
  explicit accept/reject review and golden-dataset application that fails on
  silent overwrites. The [offline walkthrough](docs/agent-eval-workflow.md) shows
  how to review and apply proposed cases.
- End-to-end demonstration and release gate: `examples/eval_demo.py` with
  committed versioned datasets (`eval_demo_cases_v1.json`,
  `eval_demo_cases_v2.json`) shows a quality-preserving token optimization
  from duplicated to single-fetch tool calls, driven by trace evidence through
  public interfaces only. `tests/test_eval_demo.py` fails the baseline on the
  golden budgets, passes the optimized agent, and rejects the reversed
  comparison.

### Upgrade from the prototype

Use `from langchain_ollie import RecordingSession` for the current API. This is an
intentional API break. Old handlers, prototype modules and legacy
LangChain integrations are removed rather than shimmed. Use `RecordingSession`
and attach `recording.handler` through the runtime callback config. Export with
`recording.export(path)` after consuming the invocation/stream. Do not assume the
prototype's payload capture, thread lanes, CPU attribution or implicit backend
configuration still exists. No environment file or global handler is installed.
Follow the [recording quickstart](README.md#quickstart-trace-a-langgraph-agent) for
the replacement API and [session ownership](README.md#session-resource-ownership)
when supplying your own store.

### Version policy

Python APIs, saved records and database schemas have different compatibility
boundaries. Check the relevant version before reusing an artifact after an upgrade.

Public Python APIs are experimental during 0.x; incompatible changes require
release notes. Neutral JSON bundles and Perfetto `otherData` carry a schema version;
SQLite independently uses `PRAGMA user_version=1`. Unknown SQLite versions fail
closed. A future migration must be explicit, tested and backed up; no downgrade or
automatic conversion of unrelated databases is supported. Framework claims stay
within the narrow minor ranges and required real-agent CI matrix in `pyproject.toml`.
