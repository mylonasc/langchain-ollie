# Architecture

Local-first execution tracing for modern LangGraph and Deep Agents. Perfetto is the
primary output; SQLite, process counters and OpenTelemetry are optional consumers.

## Data flow

```text
LangGraph / Deep Agents callbacks
              |
    adapter projection and redaction
              |
   recorder budgets and span lifecycle
              |
      immutable neutral records
       /                     \
  memory or SQLite        processors
       \                     /
   Perfetto JSON        OTel SDK bridge
```

## Dependency direction

- Neutral models import no adapter, store implementation or exporter.
- Stores depend only on neutral models and store-level protocols.
- Exporters depend only on neutral models, exporter protocols and optional SDKs.
- Framework adapters depend on recorder-facing interfaces, never concrete stores or
  exporters.
- The recorder coordinates stores and processors but contains no Perfetto, SQLite,
  OTLP or demo-specific behavior.
- Optional dependencies are imported only by their explicit modules, never by the
  core package at import time.

## Boundaries

- Spans express containment. Causal links require explicit evidence and are never
  inferred from names, timing or containment.
- Metadata-only capture is the default. Bounded or full payload capture is explicit,
  redacted and still subject to finite recording budgets.
- A recording belongs to one process and monotonic clock domain. Cross-process
  monotonic alignment and full distributed context propagation are not provided.
- Process resource counters are process-wide, not per-agent attribution.
- Automatic profiler timeline correlation is an evidenced no-go in
  [ADR-001](adr/001-profiler-correlation.md). The OTel bridge is export-only per
  [ADR-002](adr/002-otel-export-only.md).

Directory-level contracts are in
`src/langgraph_instrumentation/adapters/AGENT.md`,
`src/langgraph_instrumentation/stores/AGENT.md` and
`src/langgraph_instrumentation/exporters/AGENT.md`.