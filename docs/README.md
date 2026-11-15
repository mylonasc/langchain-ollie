# Documentation

Ollie helps you understand agent execution locally: capture
a bounded trace, inspect where elapsed time and model usage went, and turn those
observations into repeatable evaluation checks.

## Start with a recording

The [quickstart](../README.md#quickstart-trace-a-langgraph-agent) produces a trace
without API keys or model-provider calls. Next, learn to
[read the Perfetto timeline](../README.md#view-locally) and choose
[safe capture limits](../README.md#safety-and-recording-limits).

## Evaluate and improve an agent

Use the [evaluation guide](evals.md) to define cases, check answers and execution
budgets, and compare candidate runs with a baseline. The
[reviewed dataset workflow](agent-eval-workflow.md) shows how to turn findings into
new regression cases without silently changing the golden dataset.

## Understand and extend the library

The [architecture overview](architecture.md) explains the recording pipeline and
its boundaries. The [class and interaction diagrams](uml.md) show how sessions,
adapters, stores, processors and evaluators collaborate. Read these before adding
a custom integration.

## Choose storage and optional integrations

The default memory store is enough to create local traces. Use
[SQLite](../README.md#durable-local-recording) to reopen recordings later,
[process counters](../README.md#optional-process-resource-counters) to observe
process-wide CPU and memory, or the
[OpenTelemetry bridge](../README.md#optional-opentelemetry-export-only-bridge) to
send completed spans to an explicitly configured exporter.

## Operate safely and assess overhead

Read the [security guidance](../SECURITY.md) before capturing or sharing sensitive
data. The [recording benchmark](../benchmarks/REF04_BASELINE.md) and
[resource sampler measurements](../benchmarks/REF09_RESOURCES.md) explain overhead
and how to reproduce it. See the [changelog](../CHANGELOG.md) for migration and
compatibility guidance.

## Understand design decisions

Decision records explain limits that may affect your integration:

- [Profiler correlation](adr/001-profiler-correlation.md): why a separate sampled
  profile cannot be reliably aligned with execution spans.
- [OpenTelemetry export](adr/002-otel-export-only.md): why exporting local spans
  does not provide distributed context propagation.

For code changes, follow the [development verification steps](../README.md#development-and-artifact-verification)
and [repository contributor guide](../AGENTS.md). Maintainers can use the
[publishing guide](publishing.md) to release validated packages through PyPI
trusted publishing.
