# LangGraph Instrumentation

Safe, bounded **local agent performance traces**, viewed offline in
[Perfetto](https://ui.perfetto.dev). No server, provider credentials, or monitoring
backend is required. Callback durations describe elapsed execution, not CPU time.

Node wall duration includes callback overhead, scheduling, child work and waits;
overlapping durations must not be summed as CPU time or advertised as a proven
critical path. No automatic cross-process tracing, host tool sandbox, background
export service or full distributed-context propagation is installed.

## Quickstart: trace a LangGraph agent

From a checkout of this repository, install dependencies (Python 3.12–3.14):

```bash
uv sync --group dev --group agents
```

Save this as `quickstart.py`. It builds a minimal LangGraph chat agent with a
scripted chat model, so you can try tracing without API keys or network calls:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_instrumentation import RecordingSession

model = FakeListChatModel(responses=["Hello! I am a traced LangGraph agent."])


def respond(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("respond", respond)
graph.add_edge(START, "respond")
graph.add_edge("respond", END)
agent = graph.compile()

with RecordingSession() as recording:
    result = agent.invoke(
        {"messages": [("user", "Hello!")]},
        config={"callbacks": [recording.handler]},
    )

recording.export("trace.json")
print(result["messages"][-1].content)
```

Run it and inspect the trace:

```bash
uv run --no-sync python quickstart.py
```

1. Open [ui.perfetto.dev](https://ui.perfetto.dev) and choose **Open trace file**.
2. Select the generated `trace.json`.
3. Expand the timeline tracks and zoom into the graph invocation. You should see
   the graph, the `respond` node, and its chat-model call as nested operations.
4. Click a slice to inspect its duration and recorded attributes in the details
   panel. Use these timings to see where your agent spends elapsed time.

These are **execution traces**, not captured console logs. Prompt/response bodies
are not recorded by default. For private or air-gapped inspection, use the
[offline Perfetto UI](https://perfetto.dev/docs/visualization/offline).

For an existing LangGraph agent, the integration is just the `RecordingSession`
scope, `config={"callbacks": [recording.handler]}` on invocation, and
`recording.export(...)` afterward. Your agent and model construction stay unchanged.

## More runnable examples

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

## Evaluations

The evaluation layer runs local cases under `RecordingSession` and derives metrics
from the same neutral traces. It is not a second tracing implementation and does
not require a provider, server or database.

```python
from langgraph_instrumentation import (
    EvalCase,
    EvalRunner,
    EvalTurn,
    ExactText,
    LangGraphEvalAdapter,
)

case = EvalCase(
    case_id="add",
    turns=(EvalTurn.user("Add 2 and 3", checks=(ExactText("Result: 5"),)),),
)
run = EvalRunner(LangGraphEvalAdapter(app)).run([case])
assert run.passed
```

`EvalRunner` creates a fresh recording session and store per case trial, so cases
cannot observe each other's traces or conversation state. `LangGraphEvalAdapter`
attaches the callback through the runtime config and can either accumulate message
history or use a per-run checkpoint `thread_id`. Turn checks run after each turn;
case checks run after the conversation. JSON, JSONL and CSV datasets load through
`load_cases`/`save_cases`.

Every metric records its source, unit, scope and completeness. `tokens.*` comes
from provider-reported usage: unavailable usage is `missing`, never a silent zero.
Process CPU/RSS measurements remain process-wide and are not per-agent attribution.
Run the offline example with:

```bash
uv run --no-sync python examples/local_eval.py
```

Repeated trials aggregate with `aggregate_run`
(fail-safe `WORST` by default; explicit `BEST`/`MAJORITY` and `MEAN`/`MEDIAN`
metric policies), and `compare_runs` separates quality from explicit `MetricRule`
performance gates with tolerance, direction and missing-metric policy. Structured
state checks use `StatePredicate`; optional judges adapt via `JudgeEvaluator`
without adding a provider dependency.

Runs, aggregates and comparisons persist as canonical JSON through `save_run`,
`save_aggregate`, `save_comparison` (or mixed JSONL via `save_artifacts`) and
reopen with `load_run`, `load_aggregate`, `load_comparison`, `load_any` and
`load_artifacts`. Artifacts reference traces by ID and never embed payloads.
The same operations are available offline from the command line, including a
static HTML report and CI-friendly exit codes:

```bash
uv run --no-sync python -m langgraph_instrumentation evals run cases.json \
  --agent mymodule:build_app --out run.json
uv run --no-sync python -m langgraph_instrumentation evals compare base.json run.json \
  --rule "tokens.total:lower_is_better:10:0.05"
uv run --no-sync python -m langgraph_instrumentation evals report run.json --out report.html
```

See `docs/evals.md` for the artifact schema, CLI reference and exit codes.

### End-to-end demonstration

`examples/eval_demo.py` runs the complete loop offline with scripted models:
a baseline agent answers correctly but fetches every fact twice; trace evidence
shows the duplicated `lookup` calls; an external coding-agent step proposes a
single-fetch regression case through the public API; after explicit review the
golden dataset grows from `eval_demo_cases_v1.json` to
`eval_demo_cases_v2.json`; an optimized agent passes the golden suite with
lower token consumption while answers stay identical.

```bash
uv run --no-sync python examples/eval_demo.py --out-dir /tmp/eval-demo
```

Limits: scripted token counts stand in for provider usage, one redundant-fetch
pattern stands in for real bottlenecks, and the "external agent" step is a
deterministic script over public interfaces rather than an autonomous loop.
`tests/test_eval_demo.py` pins the gate: the baseline fails the golden
budgets, the optimized agent passes, and the reversed comparison regresses.

### Agent-facing inspection and evaluation workflows

Coding agents work through bounded, read-only summaries that never require
internal modules or database access:

```python
from langgraph_instrumentation import summarize_failures, turn_evidence

summary = summarize_failures(run)  # failed trials, reasons, trace IDs
evidence = turn_evidence(run, "add", 0, traces=bundles)  # calls, metrics, output
```

Candidate dataset changes are explicit `EvalProposal` records with rationale,
evidence references and fully specified cases. They validate structurally,
record an accept/reject `ReviewRecord` with reviewer provenance, and only then
merge into golden datasets via `apply_proposal` — collisions fail instead of
overwriting silently. The packaged walkthrough in
`docs/agent-eval-workflow.md` runs the full loop offline using only public
interfaces; no MCP server, agent runtime or hosted service is introduced.

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
task. Step/path/namespace/checkpoint metadata is retained where supplied. Root
status describes the invocation's outcome: a handled failed retry does not turn a
successful root into an error, and failed attempt spans remain visible. Runtime
interrupts are control flow, not application errors; `execution.interrupted` marks
the invocation.

A resumed invocation is a new trace, not a span that pretends to survive restart.
Link it explicitly to the prior root:

```python
from langgraph_instrumentation import SpanLink

link = SpanLink(previous_root.trace_id, previous_root.span_id, "resumes")
config = {
    "callbacks": [recording.handler],
    "configurable": checkpoint_config,
    "metadata": {"local_trace.links": [link.to_dict()]},
}
```

Links never imply causality from names, containment, or timing. `link:*` arrows
render only when both endpoints exist in the same exported clock domain; external
endpoints remain metadata. Unknown or evicted parents become explicit partial roots
in default mode and are rejected in strict mode.

## Safety and recording limits

Metadata-only capture is the default: bodies are replaced with original types and
sizes without traversing builtin container contents. Exceptions retain only their
type; internal diagnostics never log exception bodies or tracebacks.

```python
from langgraph_instrumentation import CapturePolicy, RecordingLimits, RecordingSession

with RecordingSession(
    capture=CapturePolicy.bounded(max_string_length=200),
    limits=RecordingLimits(max_spans=1000, max_events=2000),
) as recording:
    pass  # invoke your agent with recording.handler
```

Content capture is explicit and may expose sensitive text. `RedactionPolicy`
redacts matching mapping fields but is not a universal secret detector for free
text. Message content is never deduplicated across recordings. Exception message
capture requires `capture_exception_messages=True`; stack capture is not supported.
All capture modes retain safety budgets of 2,048 traversal nodes, 16 KiB encoded
payload and depth 64. `full()` removes the smaller bounded-mode limits, not these
safety budgets; truncation and budget markers are explicit.

Default recorder limits are 4,096 spans, 8,192 events and 4,096 metrics per
recording, with per-trace caps of 2,048 spans, 4,096 events and 2,048 metrics.
Attributes are capped at 16 KiB per mapping and 8 MiB globally / 2 MiB per trace.
The memory store retains 128 finalized traces. Completion is not denied when
admission is exhausted; attributes and events are dropped before structure.
`recording.diagnostics` reports loss, store retention and lane fallback; nonzero
means incomplete recording. Override limits with `RecordingLimits` and start a new
session to replenish admission budgets.

Spans older than ten minutes are cooperatively abandoned on the next recorder
access; there is no background expiration thread. Streaming records chunk counts
and offsets without retaining text unless `capture_streaming_tokens=True` is set.
Actual provider usage is taken from the final model response exactly once. See
[`benchmarks/REF04_BASELINE.md`](benchmarks/REF04_BASELINE.md) for overhead and
regression thresholds.

Direct recorder/store calls take trusted JSON, not arbitrary application payloads.
Apply capture policy before that boundary. Direct store writes do not inherit
recorder budgets.

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
transactions, WAL and a serialized connection; it stores neutral records, links,
resource identity, events and metrics, not Perfetto dictionaries. Session loss
diagnostics persist on orderly close. `TraceQuery` bounds listing reads;
`delete_traces(before_unix_ns=...)` removes whole finalized traces only. Exact
start/completion replays are idempotent; unknown schemas are never migrated
silently.

After a crashed writer, normal open preserves unfinished spans. **Only after all
previous writers have stopped**, opt into `SQLiteTraceStore(path, recover=True)`
or `python -m langgraph_instrumentation recover agent.db`. Recovery marks work
abandoned and its duration as a last-observed lower bound. Do not reuse a nonempty
database with a new `RecordingSession`; reopen it for query/export or start a new
recording file. Prefer memory capture when minimizing callback I/O overhead.

### Session resource ownership

The session owns and closes its default memory store. A supplied `store=` must be
empty and dedicated to this recording; it remains open after session shutdown.
Snapshots are immutable and remain exportable after close. `close()` is idempotent.
Path exports use atomic replacement and UTF-8; file-object exports cannot guarantee
atomicity. The destination's parent directory must already exist. Lower-level
`TraceRecorder` closes its supplied store by default; set `owns_store=False` to
retain ownership.

Flush/close timeouts are cooperative, not hard cancellation. Preserve the original
application exception when instrumentation also fails. Strict mode is for tests and
diagnostics, not the default production policy. Accepted in-memory lifecycle is
independent of destinations: if a store operation fails, correlation continues and
independent `processors=` still receive completed spans. The failing store may
contain a partial recording; diagnostics disclose loss, not durability. At most 256
lifecycle operations may queue behind a slow store. Event/metric appends are not
automatically retried after an uncertain acknowledgement; a remote commit can
succeed before its response is lost.

## Supported scope and migration

The locked/minimum baseline is `langchain-core==1.8.0`, `langgraph==1.2.21`, and
`deepagents==0.7.38`. CI tests locked/current supported resolutions within core
1.8.x, LangGraph 1.2.x, and Deep Agents 0.7.x on Python 3.12–3.14. Deep Agents is a
development agent dependency, not a core requirement. Only tested modern-runtime
callbacks are supported; generic standalone legacy chains, classic completion LLMs,
retriever hooks, old signatures/import paths, and automatic patching are not.

**Clean break:** `PerfettoTracer`, `PerfettoLogger`, and
`LangGraphInstrumentationHandler` have been removed. Replace them with
`RecordingSession` and explicit callback attachment; there are no compatibility
wrappers or logging-handler side effects. See [`CHANGELOG.md`](CHANGELOG.md) for
upgrade notes and [`docs/architecture.md`](docs/architecture.md) for component
boundaries.

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

## Optional integrations

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
missing/dropped counts. Units/PID remain in metric metadata. CPU percent requires
at least two samples; the runnable example takes an explicit baseline and final
sample so short agents still show a delta. Counters describe the entire current
process, never an individual overlapping agent or subprocess.
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

### Scope limits and alternatives

Full distributed propagation and automatic stack timeline correlation are not
provided. [ADR-001](docs/adr/001-profiler-correlation.md) records the no-go for
automatic py-spy Speedscope alignment; no profiler, attach or privilege escalation
ships. For broader monitoring, evaluate
[OpenInference](https://github.com/Arize-ai/openinference) with local Phoenix,
MLflow Tracing or self-hosted Langfuse; this project focuses on standalone local
artifacts.
