# Changelog

## 0.1.0 — local-first API (unreleased)

- Modern LangGraph/Deep Agents callbacks, bounded metadata-first recording,
  streaming summaries, explicit causal links and isolated destinations.
- Deterministic atomic Perfetto export and transactional SQLite v1 with read-only
  inspection/export CLI and explicit stopped-writer crash recovery.
- Real offline agent coverage on standard CPython 3.12–3.14.
- Optional process-wide CPU/RSS counters and SDK-backed export-only OTel bridge.
  The profiler compatibility spike ended in an evidenced no-go for automatic
  py-spy/Speedscope timeline correlation; see ADR-001.
- Local evaluation foundation: immutable cases/trials, a sync/async execution
  adapter for LangGraph, per-trial recording isolation, trace-derived token,
  latency, tool and optional process metrics with explicit provenance, composable
  deterministic checks and time/resource budgets. JSON, JSONL and CSV datasets
  load without a provider.
- Evaluation quality slice: fail-safe repeated-trial aggregation, baseline
  comparison with explicit metric rules/tolerances/directions, structured state
  predicates and an optional provider-free judge protocol. Execution, assessment
  and missing-evidence outcomes stay distinct.
- Agent-facing inspection and proposal workflows: bounded read-only failure,
  turn and regression summaries over run artifacts plus caller-supplied trace
  bundles (missing evidence explicit, truncation flagged); feedback records
  with provenance and a structural privacy boundary; candidate proposals with
  rationale, evidence refs and fully specified cases; structural validation,
  explicit accept/reject review and golden-dataset application that fails on
  silent overwrites. A packaged offline walkthrough uses only public
  interfaces. No MCP server, agent runtime or hosted service is introduced.

### Upgrade from the prototype

This is an intentional API break. Old handlers, prototype modules and legacy
LangChain integrations are removed rather than shimmed. Use `RecordingSession`
and attach `recording.handler` through the runtime callback config. Export with
`recording.export(path)` after consuming the invocation/stream. Do not assume the
prototype's payload capture, thread lanes, CPU attribution or implicit backend
configuration still exists. No environment file or global handler is installed.

### Version policy

Public Python APIs are experimental during 0.x; incompatible changes require
release notes. Neutral JSON bundles and Perfetto `otherData` carry a schema version;
SQLite independently uses `PRAGMA user_version=1`. Unknown SQLite versions fail
closed. A future migration must be explicit, tested and backed up; no downgrade or
automatic conversion of unrelated databases is supported. Framework claims stay
within the narrow minor ranges and required real-agent CI matrix in `pyproject.toml`.
