# LangGraph Instrumentation

Safe, bounded **local agent performance traces**, viewed offline in
[Perfetto](https://ui.perfetto.dev). No server, provider credentials, or monitoring
backend is required. Callback durations describe elapsed execution, not CPU time.

Node wall duration includes callback overhead, scheduling, child work and waits;
overlapping durations must not be summed as CPU time or advertised as a proven
critical path. No automatic cross-process tracing, host tool sandbox, background
export service or full distributed-context propagation is installed.

## Install and run

Python 3.12–3.14, standard GIL-enabled CPython:

```bash
uv sync --group dev --group agents
uv run --no-sync python examples/local_agent.py trace.json
```

The example executes a real LangGraph tool-calling loop with a scripted model and
local addition tool. It prints `Result: 5` and writes a Perfetto artifact. It never
calls an LLM provider.

### Deep Agents, without a provider

```bash
uv sync --group dev --group agents
uv run --no-sync python examples/local_deep_agent.py deep-agent-trace.json
```

This runs a real `deepagents.create_deep_agent` with scripted coordinator and
worker models. It writes and reads `/report.txt` in a **virtual state filesystem**,
delegates via `task`, and executes a local multiplication tool in the worker.
There are no host-file or shell tools and no model-provider fallbacks. The result
is `Result: 10. Report written and read; worker completed.` The virtual report is
returned in agent state; no `/report.txt` is written to your computer.

`examples/scripted_model.py` implements tool binding, explicit sync/async methods,
fragmented tool-call streaming, and usage emitted once per model call. It is
example/test support, not part of the library API. Both examples use real agent
runtimes and the same public recording API. User applications can attach it to
their own Deep Agents through the callback config shown below.

For your own application, install the package with your modern agent runtime:

```python
from langgraph_instrumentation import RecordingSession

with RecordingSession() as recording:
    result = app.invoke(inputs, config={"callbacks": [recording.handler]})

# Snapshot and export remain usable after close.
recording.export("trace.json")
```

The same callback works with `await app.ainvoke(...)`, `app.stream(...)`, and
`app.astream(...)`. Finish consuming streams before closing a session. Deep Agents
e2e tests cover all four methods, nested worker chunks, token accounting, privacy,
and repeated runs sharing the callback. Per-token event capture stays disabled by
default. Use one session per bounded recording window. Concurrent invocations
sharing a handler retain separate trace IDs and parentage.

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

### Graph identity, retries and checkpoints

Callback UUIDs identify individual executions; node names alone do not identify a
task. Step/path/namespace/checkpoint metadata is retained where supplied by the
supported runtime. Nested and dynamic fan-out executions keep separate canonical
span IDs and overlap-safe visual tracks. Root status describes the invocation's
outcome: a handled failed retry does not turn a successful root into an error.
Failed attempt spans remain visible. Runtime interrupts are control flow, not
application errors; `execution.interrupted` marks the interrupted invocation.

A resumed invocation is a new trace, not a span that pretends to survive restart.
Use a checkpoint plus an **explicit evidenced link** to the prior root:

```python
from langgraph_instrumentation import SpanLink

link = SpanLink(previous_root.trace_id, previous_root.span_id, "resumes")
config = {
    "callbacks": [recording.handler],
    "configurable": checkpoint_config,
    "metadata": {"local_trace.links": [link.to_dict()]},
}
```

The same API supports other explicitly known `follows_from`/dependency links.
`TraceRecorder.start_span(links=...)` accepts neutral `SpanLink` records (at most
32 retained). Links round-trip independently from parents. `link:*` arrows render
only when both endpoints exist in the same exported clock domain and are ordered
in time; external endpoints remain metadata. Graph dependencies are **not inferred
from names, containment, or join timing**. Thread/session IDs group invocations but
alone do not prove a resume relationship. Cross-process clock correlation is not
supported by the local exporter.

Unknown/evicted parents in default mode create a separate explicitly partial root
with `recording.orphan_parent` and a loss counter, never attach to an unrelated
trace. Strict mode rejects unknown parents. Late children use the bounded completed
context cache only while their parent's context remains known.

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
256 lanes, and 16 KiB per attribute mapping. Aggregate encoded attributes are
capped at 8 MiB globally and 2 MiB per trace, including repeated completion data
(conservative charging). Structural record counts/names bound additional Python
overhead; these are not exact RSS limits. Per-trace caps are 2,048 spans, 4,096
events, and 2,048 metrics; each span retains at most 128 detailed events. At most
128 spans may be active. Names/IDs are bounded. Completion is
not denied when admission is exhausted. The memory store retains up to 128 traces,
evicting only finalized traces. Start a new session to replenish admission budgets.
`recording.diagnostics` returns loss observations, including store retention;
nonzero means incomplete
recording, including track/lane fallback. Attributes/events are dropped before
structure; admission never evicts active traces. Completed numeric budget ledgers
stay bounded by lifetime span admission so late children cannot reset a budget.
Span payloads leave active state at completion; context caches retain only headers.

Spans older than ten minutes are cooperatively abandoned on new admission,
stream-chunk observation, snapshot, or explicit `recorder.expire_spans()`. There is
no background expiration thread: idle recorders are maintained when accessed or
closed. Configure `max_span_age_ns` for longer operations. Late ends of expired
spans are harmless; loss and abandoned status remain visible.

Streaming records first/last chunk offsets and chunk counts in `llm.streaming`
without retaining text. These are callback chunks, not tokenizer token counts.
`RecordingSession(capture_streaming_tokens=True)` additionally retains detailed
events under the same global/per-trace/per-span budgets. Actual provider usage is
still taken from the final model response exactly once. See
[`benchmarks/REF04_BASELINE.md`](benchmarks/REF04_BASELINE.md) for overhead and
regression thresholds.

Direct recorder/store calls take trusted JSON and safe operation/resource identity,
not arbitrary application payloads. Apply capture policy before that boundary.
Direct store writes do not inherit recorder budgets.

## Lifecycle and ownership

### Durable local recording

Runnable no-network close/reopen example (use a new database path):

```bash
uv run --no-sync python examples/local_sqlite.py agent.db sqlite-trace.json
```

```python
from langgraph_instrumentation import RecordingSession, SQLiteTraceStore

store = SQLiteTraceStore("agent.db")  # new dedicated recording database
with RecordingSession(store=store) as recording:
    result = app.invoke(inputs, config={"callbacks": [recording.handler]})
store.close()
```

Later, from another process:

```bash
python -m langgraph_instrumentation list agent.db --limit 20 --offset 0
python -m langgraph_instrumentation summary agent.db TRACE_ID
python -m langgraph_instrumentation export agent.db TRACE_ID trace.json
```

CLI inspection uses read-only connections. The store uses versioned schema v1,
foreign-key cascades, WAL, FULL synchronization, per-operation transactions and a
two-second busy timeout. One connection is serialized across threads; independent
connections coordinate through SQLite. It stores neutral records, links, resource
identity, events and metrics, not Perfetto dictionaries. Session loss diagnostics
persist on orderly close. `TraceQuery(limit=..., offset=...)` bounds listing reads;
`delete_traces(before_unix_ns=...)` retains active traces and deletes whole finalized
traces. Exact start/completion replays are idempotent. No silent migration of an
unknown schema or unrelated unversioned database is attempted.

After a crashed writer, normal open preserves unfinished spans. **Only after all
previous writers have stopped**, opt into `SQLiteTraceStore(path, recover=True)`
or `python -m langgraph_instrumentation recover agent.db`. Recovery marks work
abandoned and flags its duration as a last-observed lower bound; it never treats
offline downtime as execution or guesses that another live writer is dead.
Do not reuse a nonempty database with a new `RecordingSession`; reopen it for
query/export, or start a new recording file. Database retention/rotation is caller
controlled. Prefer memory capture when minimizing callback I/O overhead; durable
SQLite commits have a real latency cost.

### Session resource ownership

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

Accepted in-memory lifecycle is independent of destinations in default mode.
If a store start/event/completion fails, correlation continues and independent
`processors=` still receive completed immutable spans (including resource/service
identity). The failing store may contain a partial recording; diagnostics disclose
loss, not durability. Detailed event failure stops writes for that span rather than
attempting a misleading completion against incomplete persisted events. Strict
mode instead rejects failed transitions and permits explicit diagnostic retries.
Destination diagnostics are body-free and rate-limited.

At most 256 lifecycle operations may queue behind a slow store. Caller wait
timeouts do not forcibly cancel uncooperative Python callbacks; release or stop
such a destination to complete background shutdown. Callbacks must not drive their
own recorder lifecycle; processor reentry is rejected. Concurrent session close
shares recorder shutdown and serializes final snapshots/resource cleanup. A failed
best-effort close has a stable `False` result and does not redispatch spans; final
snapshots remain readable once resources have quiesced.

Exact store span start/completion replays are idempotent; conflicting replays fail.
Event/metric appends lack operation IDs and are **not** automatically retried after
an uncertain acknowledgement. A remote commit can succeed before its response is
lost: the store protocol does not promise that every remote exception proves rollback.

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
uv sync --group dev --group agents --group resources --group otel
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

Required CI covers minimum, locked and current supported framework versions on
Python 3.12–3.14, plus clean-wheel no-network graph, Deep Agents and SQLite/CLI
quickstarts outside the checkout. Free-threaded CPython and other runtimes/platforms
are not claimed by this matrix. See [release/upgrade notes](CHANGELOG.md),
[security and artifact handling](SECURITY.md), and the
[benchmark reproduction and regression budgets](benchmarks/REF04_BASELINE.md).

## Roadmap and alternatives

### Optional process resource counters

Install `pip install 'langgraph-instrumentation[resources]'` or
`uv sync --group dev --group agents --group resources`, then run:

```bash
uv run --no-sync python examples/local_resources.py resources.json
```

`from langgraph_instrumentation.resources import ProcessSampler` is opt-in; no
sampler thread or psutil import occurs on default library import. Use
`with ProcessSampler(recording.recorder, interval=0.1, max_samples=600): ...` inside
the recording scope. A separate **process** trace contains CPU seconds, RSS bytes,
CPU percent (one logical core = 100%, potentially over 100%), read overhead and
missing/dropped counts. Units/PID remain in metric metadata. Counters describe
the entire current process, never an individual overlapping agent or subprocess.
The sampler has no extra sample buffer, stops at its attempt cap, shares recorder
budgets, and must close before the recording. Source calls must return promptly;
a timed-out close cannot forcibly interrupt a native/custom source call.

### Optional OpenTelemetry export-only bridge

Install `pip install 'langgraph-instrumentation[otel]'`. The no-network SDK demo:

```bash
uv sync --group dev --group agents --group resources --group otel
uv run --no-sync python examples/local_otel.py otel-local.json
uv run --no-sync pytest -m otel
```

Use `OpenTelemetryProcessor(exporter)` from
`langgraph_instrumentation.exporters.otel` in `RecordingSession(processors=[...])`.
Supply an explicit trusted SDK SpanExporter (for example OTLP/HTTP with a chosen
endpoint); the bridge owns it. Default batching uses the SDK's bounded queue.
No SDK import, global provider, exporter environment configuration, or network
destination is installed by the local path. SDK exporter transports own retries.

This is **export-only**, not full context interoperability: canonical local IDs,
parents, links/resources and Unix timing survive, but incoming remote context,
flags/tracestate, SDK sampling and outgoing/manual-child propagation do not.
Converted spans are sampled INTERNAL spans; richer kinds/statuses stay namespaced.
No GenAI/OpenInference semantic compliance or metric export is claimed. Read
[ADR-002](docs/adr/002-otel-export-only.md) for mapping and lifecycle/loss limits.
`force_flush(seconds)` is bounded by SDK policy; shutdown bounds the caller's wait
but cannot kill an uncooperative foreign exporter. A destination failure does not
erase local Perfetto records. The optional suite uses only an isolated loopback
OTLP receiver, never an external service; the default suite excludes it.

See [`EPIC_REFACTOR.md`](EPIC_REFACTOR.md) and [epic #13](https://github.com/mylonasc/langgraph_instrumentation/issues/13).
Local SQLite, explicit graph links, bounded recording and modern agent e2e are
implemented. Process CPU/RSS and SDK export-only OTel are optional extras; full
distributed propagation and automatic stack timeline correlation are not provided.

Profiler spike outcome: [ADR-001](docs/adr/001-profiler-correlation.md) is a no-go
for automatic py-spy Speedscope-to-Perfetto alignment: its sample-count timeline
lacks the timestamps needed to recover elapsed gaps. The ADR includes an external
opt-in companion-view workflow; no profiler, attach or privilege escalation ships.

For broader monitoring now, evaluate [OpenInference](https://github.com/Arize-ai/openinference)
with local Phoenix, MLflow Tracing, or self-hosted Langfuse. OpenInference can target
any OTel-compatible backend; this project's focus is standalone local artifacts.
