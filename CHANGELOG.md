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
- Portable evaluation artifacts and local inspection: canonical JSON run,
  aggregate and comparison artifacts (plus mixed JSONL), a thin CLI over the
  public Python API with machine-readable output and CI gate exit codes, and a
  static offline HTML report. Agent-facing workflows remain in the indexed
  evaluation slices.

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
