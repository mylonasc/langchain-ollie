# Changelog

## 0.1.0 — local-first API (unreleased)

- Modern LangGraph/Deep Agents callbacks, bounded metadata-first recording,
  streaming summaries, explicit causal links and isolated destinations.
- Deterministic atomic Perfetto export and transactional SQLite v1 with read-only
  inspection/export CLI and explicit stopped-writer crash recovery.
- Real offline agent coverage on standard CPython 3.12–3.14.

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
