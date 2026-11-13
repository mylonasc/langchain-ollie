# LangGraph Instrumentation

Safe, bounded **local agent performance traces**, viewed offline in
[Perfetto](https://ui.perfetto.dev). No server, provider credentials, or monitoring
backend is required. Callback durations describe elapsed execution, not CPU time.

## Install and run

Python 3.12–3.14, standard GIL-enabled CPython:

```bash
uv sync --group dev --group agents
uv run --no-sync python examples/local_agent.py trace.json
```

The example executes a real LangGraph tool-calling loop with a scripted model and
local addition tool. It prints `Result: 5` and writes a Perfetto artifact. It never
calls an LLM provider. Deep Agents runtime coverage is the next slice, REF-03.

For your own application, install the package with your modern agent runtime:

```python
from langgraph_instrumentation import RecordingSession

with RecordingSession() as recording:
    result = app.invoke(inputs, config={"callbacks": [recording.handler]})

# Snapshot and export remain usable after close.
recording.export("trace.json")
```

The same callback works with `await app.ainvoke(...)`. Finish consuming streams
before closing a session. Use one session per bounded recording window; shared
handler invocation semantics are being expanded in REF-05.

## View locally

Open Perfetto and use **Open trace file**. For sensitive data or air-gapped work,
use the [self-hosted/offline Perfetto UI](https://perfetto.dev/docs/visualization/offline)
instead of depending on the hosted viewer. No upload/export destination is
configured by this library.

Tracks separate overlapping operations, even when a node name is repeated.
`contains` arrows show **span containment**, not graph transitions, data flow,
or a proven critical path. Token usage appears as counters. The JSON's `otherData`
contains schema version, process/clock identity, resource metadata, loss counters,
and an `incomplete` flag. Unfinished spans become abandoned on orderly shutdown.

## Safety and recording limits

Metadata-only capture is the default: bodies are replaced with original types/
sizes without traversing builtin container contents. Arbitrary mappings report
their type, not a possibly unsafe custom length. Exceptions retain their type
only. Internal diagnostics do not log exception bodies/tracebacks.

```python
from langgraph_instrumentation import CapturePolicy, RecordingLimits, RecordingSession

with RecordingSession(
    capture=CapturePolicy.bounded(max_string_length=200),
    limits=RecordingLimits(max_spans=1000, max_events=2000),
) as recording:
    pass  # invoke your agent with recording.handler
```

Content capture is explicit and may expose sensitive text. `RedactionPolicy`
redacts matching mapping fields; it is not a universal secret detector for free
text. Message content is never deduplicated across recordings. Exception message
capture additionally requires the adapter's `capture_exception_messages=True`.
Stack capture is not supported.

All capture modes have safety budgets: 2,048 traversal nodes, 16 KiB encoded payload
and a maximum depth of 64. `full()` disables the smaller bounded-mode collection/
string limits, not these safety budgets. Truncation/budget markers are explicit.

Default lifetime recorder budgets: 4,096 spans, 8,192 events, 4,096 metric points,
256 lanes, and 16 KiB per attribute mapping. Names/IDs are bounded. Completion is
not denied when admission is exhausted. The memory store retains up to 128 traces,
evicting only finalized traces. Start a new session to replenish admission budgets.
`recording.diagnostics` returns loss observations, including store retention;
nonzero means incomplete
recording, including track/lane fallback. REF-04 adds pressure-aware retention.

Direct recorder/store calls take trusted JSON and safe operation/resource identity,
not arbitrary application payloads. Apply capture policy before that boundary.
Direct store writes do not inherit recorder budgets.

## Lifecycle and ownership

The session owns and closes its default memory store. A supplied `store=` must be
empty and dedicated to this recording; it remains open after session shutdown.
Session snapshots are immutable, and its final snapshot remains exportable after
the owned store closes. `close()` is idempotent. Exporting to a path uses atomic
replacement and UTF-8; exporting to a text file object cannot guarantee atomicity.
The destination's parent directory must already exist.

Lower-level `TraceRecorder` closes its supplied store by default; set
`owns_store=False` to retain ownership. Flush/close timeout arguments are seconds;
dependency timeouts are cooperative, not hard cancellation. Preserve your original
application exception when instrumentation also fails. Strict mode is for tests
and diagnostics, not the default production failure policy.

## Supported scope and migration

The locked/minimum baseline is `langchain-core==1.8.0`, `langgraph==1.2.21`, and
`deepagents==0.7.38`. CI tests locked/current supported resolutions within core
1.8.x, LangGraph 1.2.x, and Deep Agents 0.7.x on Python 3.12–3.14. Deep Agents is a
development agent dependency, not a core requirement. Only tested modern-runtime
callbacks are supported; generic standalone legacy chains, classic completion LLMs,
retriever hooks, old signatures/import paths, and automatic patching are not.

**Clean break:** `PerfettoTracer`, `PerfettoLogger`, and
`LangGraphInstrumentationHandler` have been removed. Replace them with
`RecordingSession` and explicit callback attachment. Provider-backed prototype
examples were replaced with the reproducible local agent. There are no compatibility
wrappers, logging-handler side effects, or hash-derived Python thread tracks.

## Development and artifact verification

```bash
uv sync --group dev --group agents
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv build --out-dir dist
```

Default tests block sockets, remove provider credentials, and disable external
tracing. Fake only external models/tools, not the agent runtime. Perfetto importer
tests run in a separate required CI job. Provision its pinned checksum-verified
Linux x86-64 binary explicitly (this step uses the network), then run offline:

```bash
uv run --no-sync python scripts/install_perfetto.py /tmp/trace_processor_shell
PERFETTO_TRACE_PROCESSOR=/tmp/trace_processor_shell uv run --no-sync pytest -m perfetto
```

The tests themselves never download binaries. They execute real sync/async agents,
ingest artifacts, and assert slices, counters, flows, and importer error statistics.

## Roadmap and alternatives

See [`EPIC_REFACTOR.md`](EPIC_REFACTOR.md) and [epic #13](https://github.com/mylonasc/langgraph_instrumentation/issues/13).
SQLite, graph dependency links, expanded Deep Agents e2e and release hardening
follow the initial local workflow. CPU/RSS counters, profiler correlation, and
OpenTelemetry monitoring are lower-priority opt-in follow-ups, not core requirements.

For broader monitoring now, evaluate [OpenInference](https://github.com/Arize-ai/openinference)
with local Phoenix, MLflow Tracing, or self-hosted Langfuse. OpenInference can target
any OTel-compatible backend; this project's focus is standalone local artifacts.
