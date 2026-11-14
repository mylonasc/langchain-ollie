# ADR-002: optional neutral-record OpenTelemetry SDK consumer

Status: accepted for REF-11, 2026-11-14. **Export-only**, not full distributed
context interoperability. Import explicitly from `exporters.otel` with the `otel`
extra. Do not configure a global provider or read OTEL environment configuration.

## Alternatives and decision

An SDK-backed callback adapter could inherit SDK context, sampling and propagation,
but would couple local capture/lifecycle to an SDK/provider and require careful
incoming-parent/async context handling. Replaying `Tracer.start_span` would also
generate new IDs and sample again. Instead, convert completed neutral records to
SDK `ReadableSpan` and hand them to a standard `BatchSpanProcessor` (or synchronous
`SimpleSpanProcessor` for tests). The SDK owns bounded queues, export scheduling
and exporter behavior; standard OTLP exporters own transport/retry. No homemade
batch/retry queue or OTLP encoder. Parentage, IDs, links and Unix timestamps survive
conversion. Perfetto never imports this module or changes behavior.

## Context and sampling contract

- IDs are the local recording IDs. Parent context is a **local** parent in the same
  trace. Cross-trace causal references remain OTel Links with relationship metadata.
- All admitted local records are exported sampled; SDK sampler decisions are not
  replayed. Incoming OTel parents, remote bits, trace flags and tracestate were not
  captured by the local model and are **not** reconstructed. Converted flags are
  sampled, remote is false, tracestate is empty.
- There is no implicit incoming context extraction, outgoing injection, manual OTel
  child propagation or span activation. An independently active SDK span does not
  become this recording's parent. Applications needing full propagation should use
  an SDK-native integration instead. This limitation is tested and user-visible.
- Do not mix these records with a competing integration for the same invocation
  unless duplicate telemetry is intentional.

## Conventions (local version 1)

Preserve arbitrary local kinds/statuses in `local_trace.span.kind/status`;
OTel kind is INTERNAL, never infer remote CLIENT/SERVER from an LLM/tool name.
SUCCESS maps to OK, ERROR to ERROR, cancellation/abandonment to UNSET with explicit
local status. Do not send error descriptions by default. Events use Unix ns and
links retain canonical IDs. Span resource attributes carry service/process identity.
Nested neutral attributes are canonical JSON strings, not invalid SDK attribute
objects. `local_trace.schema.version=1` and instrumentation scope version identify
the mapping. No unpublished OTel schema URL is invented.

GenAI conventions are evolving and scripted models/tools are not necessarily
remote inference. This version deliberately claims **neither** a GenAI semantic
convention version nor OpenInference compliance: sanitized local `llm.*`, usage,
tool and capture metadata are preserved as recorded. Metrics remain local; this
span bridge does not initialize a global MeterProvider or promise metric export.
Future GenAI/OpenInference mappings need a separately versioned ADR and fixtures,
not automatic renaming that changes meanings.

## Lifecycle, loss and privacy

The bridge owns the supplied exporter. Local TraceRecorder already isolates
destinations; a rejecting exporter cannot erase local records. Its wrapper counts
exceptions/failure results without logging exception bodies. Explicit exporter
implementations may still log; configure trusted SDK/exporter logging separately.
`force_flush(seconds)` supplies the SDK millisecond timeout and returns false on known
failure. Reviewed SDK 1.45.2 ignores that timeout in its blocking batch flush, so
one shared flush worker bounds the caller wait independently; subsequent calls
wait for the same ongoing flush rather than accumulating workers. This is not
a custom telemetry queue or retry engine. SDK queue overflow follows SDK drop/metrics policy, not an exactly-once
guarantee or an authoritative local export-loss count.

SDK shutdown exposes no public timeout; one cleanup thread bounds the caller's
wait and later shutdown calls wait for that same cleanup. A foreign exporter must
return cooperatively; timeout does not kill its thread. Default recording never
creates these workers. Real SDK tests plus a loopback OTLP HTTP receiver validate
wire IDs/resources/links and failure/timeout/cleanup. No collector/network runs in
the default core suite; the optional `otel` suite explicitly permits only loopback
in an isolated subprocess. No external collector is required or contacted.
