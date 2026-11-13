# LangGraph Instrumentation

Instrumentation for understanding LangGraph agent execution locally in
[Perfetto](https://ui.perfetto.dev), with a staged refactor toward interchangeable
trace stores and observability exporters.

The current branch is implementing safe, bounded local tracing. The modular
Perfetto exporter, SQLite persistence, and optional OTLP support are tracked in
[`EPIC_REFACTOR.md`](EPIC_REFACTOR.md) and GitHub epic
[#13](https://github.com/mylonasc/langgraph_instrumentation/issues/13).

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
virtual environments, locking, and builds. Supported Python versions: 3.12–3.14.
Use standard (GIL-enabled) CPython; free-threaded builds are not supported by
the current framework dependency stack.

```bash
uv sync --group dev --group agents
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build --out-dir dist
```

Install optional example dependencies with:

```bash
uv sync --group dev --extra examples
```

The future OTLP implementation has its own optional dependency set:

```bash
uv sync --extra otlp
```

`uv.lock` is committed so local development and CI resolve the same dependency
versions.

## Current API

> **Deprecated and unsafe for sensitive data:** the prototype example below
> captures content by default and does not use the new capture policies. It
> remains only until REF-02 delivers the replacement Perfetto exporter; do not
> deploy it with application secrets or customer payloads.

```python
from langgraph_instrumentation import (
    LangGraphInstrumentationHandler,
    PerfettoLogger,
    PerfettoTracer,
)

tracer = PerfettoTracer(process_name="langgraph-agent")
logger = PerfettoLogger()
callback = LangGraphInstrumentationHandler(tracer=tracer, logger=logger)

# Pass callback through a LangGraph invocation config:
# app.invoke(inputs, config={"callbacks": [callback]})

tracer.save("trace.json")
```

Open the generated JSON file in `ui.perfetto.dev`.

The public API above is retained only while its behavior is characterized. The
epic intentionally introduces a clean-break recorder/store/exporter API rather
than permanent compatibility wrappers.

## Neutral Trace Records

The first refactor stage provides exporter-independent, immutable domain
records. Trace IDs are non-zero 128-bit values and span IDs are non-zero 64-bit
values, serialized as fixed-width lowercase hexadecimal strings.

```python
from langgraph_instrumentation import Span, SpanId, SpanKind, TraceId

span = Span(
    trace_id=TraceId(1),
    span_id=SpanId(1),
    name="Graph: demo",
    kind=SpanKind.GRAPH,
    start_time_unix_ns=1_000,
    start_time_monotonic_ns=100,
)

payload = span.to_dict()
restored = Span.from_dict(payload)
assert restored == span
```

`Trace`, `Span`, `SpanEvent`, `MetricPoint`, `TraceBundle`, `TraceSummary`, and
`TraceQuery` expose explicit `to_dict()` and `from_dict()` methods. Their output
contains only JSON-compatible values. Attribute mappings are validated and
deeply frozen when records are constructed, so later mutation of caller-owned
objects cannot change a stored trace.

Use `SystemClock` and `RandomIdGenerator` in production. `DeterministicClock`
and `DeterministicIdGenerator` support repeatable tests without sleeps or random
fixtures.

## Capture Safety

`CapturePolicy` controls whether instrumentation retains no payload, structural
metadata only, bounded content, or full content. Metadata-only capture is the
default and records sizes/types rather than prompt, response, state, or tool
bodies.

```python
from langgraph_instrumentation import CapturePolicy, MessageProjector

projector = MessageProjector(capture=CapturePolicy.metadata())
```

`Sanitizer` applies depth, string, byte, and collection bounds and safely handles
cycles and unsupported objects. `RedactionPolicy` removes common secret fields
case-insensitively and supports custom predicates. `MetadataProjector` controls
which framework metadata keys are retained, while `UsageExtractor` normalizes
common provider token-usage shapes.

Full capture is explicit and may retain sensitive application data. Redaction
still applies in full mode. Full means no bounded-mode collection/string limit,
not unlimited memory: all policies cap traversal at 2,048 nodes, encoded payload
size at 16 KiB, and full-mode recursion at 64 levels. Exceeding safety limits
produces explicit budget/truncation markers. Metadata summarizes original builtin
containers without visiting their bodies; arbitrary mappings report only a type.
Messages are never deduplicated across recordings.

The new `LangGraphCallbackHandler` records only exception types by default.
Exception message capture requires `capture_exception_messages=True` **and** a
content-enabled capture policy. Treat messages as sensitive; field redaction is
not a general secret detector for free text. Stack capture is not supported.
Internal instrumentation diagnostics omit exception bodies/tracebacks.

### Recording budgets

`TraceRecorder(limits=RecordingLimits(...))` admits at most 4,096 spans, 8,192
events, 4,096 metric points, and 256 logical lanes during its lifetime. Completion
is never denied by an exhausted admission budget. Each attribute mapping is
bounded to 16 KiB, IDs/names are bounded, and completed parent contexts contain no
span payload. `recorder.diagnostics` returns loss counters: nonzero means the
recording is incomplete (including lane fallback). Create a new recorder to
replenish budgets; automatic expiry/refill and pressure-aware retention follow
in REF-04. `MemoryTraceStore` retains at most 128 traces by default, evicting only
finalized traces. Use it behind the recorder for bounded default recording.

Direct recorder/store inputs are trusted JSON/metadata, not arbitrary framework
payloads: apply capture policies first. Resource identity and operation names must
not contain customer data. Recorder bounds do not apply to direct store writes.
Trace-level payload attributes are intentionally not accepted by `start_span`;
keep captured input on its root span, not in cached trace metadata.

### Supported framework scope

The locked baseline is `langchain-core==1.8.0`, `langgraph==1.2.21`, and
`deepagents==0.7.38`; development agent dependencies use the `agents` group.
CI tests locked and current resolutions within core 1.8.x, LangGraph 1.2.x and
Deep Agents 0.7.x on Python 3.12–3.14. These baseline versions are also the minimum
supported versions; older frameworks are not promised compatibility. Deep Agents
runtime e2e coverage arrives in REF-03; dependency installation alone is not a
compatibility claim. CI always installs the group so those tests cannot silently
skip once added. Default tests block outbound sockets and disable external tracing.

Standalone legacy LangChain chains/provider formats, old callback signatures,
automatic monkey-patching, legacy import paths, and profiling are not supported
contracts. Remaining classic callback helpers will be removed in REF-02 unless
the modern agent e2e fixture needs them. `langchain-core` is retained as the modern
agent runtime's callback/message dependency, not a legacy compatibility layer.

## Examples

Provider-backed demonstrations live under `examples/` and are not imported by
the library:

```bash
uv run --extra examples python examples/run_parallel.py
```

The parallel example expects Ollama and the configured model to be available.
The ReAct example expects an OpenAI credential. Tests never require either.

Example Perfetto output:

![Perfetto trace](src/assets/perfetto_example.png)

Corresponding graph:

![Parallel graph](src/assets/graph.png)

## Planned Architecture

The refactor separates four concerns:

- A LangGraph callback adapter translates framework callbacks.
- A recorder manages trace and span lifecycle.
- A `TraceStore` persists neutral records in memory, SQLite, or future stores.
- Exporters translate neutral records to Perfetto or OTLP.

See `EPIC_REFACTOR.md` for feature dependencies, design constraints, and
acceptance scenarios. Deferred non-blocking work is tracked in
[FRK-Techdept #12](https://github.com/mylonasc/langgraph_instrumentation/issues/12).

## OpenInference Alternative

[OpenInference](https://github.com/Arize-ai/openinference) provides automatic
OpenTelemetry instrumentation by patching LangChain/LangGraph integrations. It
is a useful alternative when local Perfetto export, explicit trace storage, or
the component boundaries planned by this project are not required.
