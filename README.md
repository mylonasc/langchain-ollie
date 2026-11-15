<h1 align="center">OLLIE: Open Local LangGraph Instrumentation &amp; Evaluation</h1>

<p align="center"><strong>Local Tracing and Evaluation for LangGraph and Deep Agents</strong></p>

Ollie is a local-first Python toolkit for tracing and evaluating agents built with
LangGraph and Deep Agents. It helps you investigate slow operations, repeated
tool calls and excessive token use, then turn those findings into repeatable
checks for correctness and execution budgets.

Explore portable execution traces visually in [Perfetto](https://ui.perfetto.dev),
define and run evaluation datasets, and compare candidate runs with a baseline
using trace-derived metrics and local HTML reports. The default workflow requires
no observability server, collector or external database. Your agent and model
construction stay unchanged.

## Why Ollie?

Agent optimization is not just about choosing a faster model. Excessive token use
can come from repeated model calls, unnecessary tool round trips, or the context
supplied at each step. Ollie brings reported token usage together with execution
structure so you can locate expensive patterns and test changes to context
engineering and agent tool design. For example, compare a tighter context strategy
or a tool that returns more focused results, then check whether token use and call
counts improve without sacrificing answer quality. Inspecting prompt or tool
content requires explicit capture; bodies are not recorded by default.

Portable visual inspection keeps the investigation close to the code: export a
trace file and open it in Perfetto to explore nested operations, overlapping work
and repeated calls without deploying a dashboard. An offline Perfetto viewer can
be used in restricted environments. Trace artifacts can still contain sensitive
metadata or explicitly captured content; review them before sharing.

Ollie grew out of a need to investigate agent behavior in environments where
hosted services, additional infrastructure and software approval processes can
slow development. It provides an MIT-licensed, artifact-first workflow for teams
that want local tracing and evaluation without deploying a broader observability
platform. Bounded recording and explicit integrations keep the setup focused on
the development loop: inspect, change, evaluate and compare. Evaluation gates
check the workloads you define; they do not guarantee production performance.

Start with the quickstart below, then choose your next task:

- [Read a timeline](#view-locally) to understand where elapsed time went.
- [Evaluate an agent](docs/evals.md) to protect answer quality and execution budgets.
- [Choose capture limits](#safety-and-recording-limits) before recording sensitive data.
- [Explore the documentation](docs/README.md) for architecture, extensions and operations.

## Quickstart: trace a LangGraph agent

Create your first trace with a local scripted model, so you can learn the recording
API without API keys or provider calls. From a checkout of this repository,
install dependencies (Python 3.12–3.14). The distribution is named
`langchain-ollie`, and Python imports use `langchain_ollie`. Use the checkout to
run the bundled examples:

```bash
uv sync --group dev --group agents
```

Save this as `quickstart.py`. It builds a minimal LangGraph chat agent with a
scripted chat model, so you can try tracing without API keys or network calls:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, MessagesState, StateGraph

from langchain_ollie import RecordingSession

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

Use these examples to see tool calls and delegated agents in a trace before
instrumenting your own application. They run on Python 3.12–3.14, standard
GIL-enabled CPython. For the design behind the API, see the
[architecture overview](docs/architecture.md) and [class diagrams](docs/uml.md).

```bash
uv sync --group dev --group agents
uv run --no-sync python examples/local_agent.py trace.json
```

The example executes a real LangGraph tool-calling loop with a scripted model and
local addition tool. It prints `Result: 5` and writes a Perfetto artifact. It never
calls an LLM provider.

### Deep Agents, without a provider

Delegation adds nested graph and tool activity. This example makes that structure
visible using a coordinator and worker that both run locally:

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

Both examples use real agent runtimes with scripted models in place of provider
calls. The scripted models are example support, not part of the library API.

For your own application, install this checkout with `pip install .` alongside
your supported agent runtime, and use your existing compiled application as `app`:

```python
from langchain_ollie import RecordingSession

with RecordingSession() as recording:
    result = app.invoke(inputs, config={"callbacks": [recording.handler]})

# Snapshot and export remain usable after close.
recording.export("trace.json")
```

The same callback works with `await app.ainvoke(...)`, `app.stream(...)`, and
`app.astream(...)`. Finish consuming streams before closing a session. Per-token
event capture stays disabled by default. Use one session per bounded recording
window; concurrent invocations sharing a handler retain separate trace IDs and
parentage. See [session ownership](#session-resource-ownership) before sharing
stores or managing shutdown yourself.

## Evaluations

Use evaluations when you want to catch wrong answers or expensive execution
patterns before they become regressions. The runner executes cases under
`RecordingSession` and derives measurements from the same traces. Given an
application `app` that answers the addition request:

```python
from langchain_ollie import (
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

`EvalRunner` creates a fresh recording session and store per case trial to isolate
traces. `LangGraphEvalAdapter` attaches the callback and isolates conversation
history by case and trial, using accumulated messages or a trial-specific
checkpoint `thread_id`. Your application remains responsible for any other state
in its models or tools. Turn checks run after each turn;
case checks run after the conversation. JSON, JSONL and CSV datasets load through
`load_cases`/`save_cases`.

Every metric records its source, unit, scope and completeness. `tokens.*` comes
from provider-reported usage: unavailable usage is `missing`, never a silent zero.
Process CPU/RSS measurements remain process-wide and are not per-agent attribution.
Run the offline example with:

```bash
uv run --no-sync python examples/local_eval.py
```

Repeat cases to reveal inconsistent behavior, then compare a candidate against
a baseline. Quality regressions always matter; performance gates apply to metrics
you explicitly choose. Save results with `save_run` for later comparison or use
the CLI with your own dataset and importable application factory:

```bash
uv run --no-sync python -m langchain_ollie evals run cases.json \
  --agent mymodule:build_app --out run.json
uv run --no-sync python -m langchain_ollie evals compare base.json run.json \
  --rule "tokens.total:lower_is_better:10:0.05"
uv run --no-sync python -m langchain_ollie evals report run.json --out report.html
```

See the [evaluation guide](docs/evals.md) for runnable cases, metric interpretation,
artifact persistence, comparison policies and CI exit codes. Saved results can
contain agent answers even when trace payload capture is disabled.

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

The baseline fails the new budgets, the optimized agent passes, and reversing
the comparison detects a regression. Scripted token counts make the result
reproducible; they are not real provider-cost measurements. The proposal step is
scripted rather than autonomous. Follow the [review workflow](docs/agent-eval-workflow.md)
to apply the same process to your own findings.

### Agent-facing inspection and evaluation workflows

Use bounded, read-only summaries to give a coding agent focused evidence rather
than an entire recording. Given a saved `run` and separately retained trace
`bundles`:

```python
from langchain_ollie import summarize_failures, turn_evidence

summary = summarize_failures(run)  # failed trials, reasons, trace IDs
evidence = turn_evidence(run, "add", 0, traces=bundles)  # calls, metrics, output
```

Candidate dataset changes are explicit `EvalProposal` records with rationale,
evidence references and fully specified cases. They validate structurally,
record an accept/reject `ReviewRecord` with reviewer provenance, and only then
merge into golden datasets via `apply_proposal` — collisions fail instead of
overwriting silently. The [dataset-improvement walkthrough](docs/agent-eval-workflow.md)
shows how to inspect results, draft a case, review it and verify the expanded suite.

## View locally

Use the timeline to identify long operations, overlapping work and repeated
calls. Open Perfetto and use **Open trace file**. For sensitive data or air-gapped work,
use the [self-hosted/offline Perfetto UI](https://perfetto.dev/docs/visualization/offline)
instead of depending on the hosted viewer. No upload/export destination is
configured by this library.

Durations measure elapsed wall time, including scheduling, waits, callback
overhead and child work. They are not CPU measurements: do not sum overlapping
slices as CPU time or treat them as a proven critical path. Use
[process counters](#optional-process-resource-counters) for process-wide resource
measurements.

Tracks separate overlapping operations, even when a node name is repeated.
`contains` arrows show **span containment**, not graph transitions, data flow,
or a proven critical path. Token usage appears as counters. The JSON's `otherData`
contains schema version, process/clock identity, resource metadata, loss counters,
and an `incomplete` flag. Unfinished spans become abandoned on orderly shutdown.

### Graph identity, retries and checkpoints

Retries and resumed runs can repeat the same node name. Use callback IDs and
parentage to distinguish individual executions; node names alone do not identify
a task. Step/path/namespace/checkpoint metadata is retained where supplied. Root
status describes the invocation's outcome: a handled failed retry does not turn a
successful root into an error, and failed attempt spans remain visible. Runtime
interrupts are control flow, not application errors; `execution.interrupted` marks
the invocation.

A resumed invocation is a new trace, not a span that pretends to survive restart.
Link it explicitly to the prior root:

```python
from langchain_ollie import SpanLink

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

Choose the smallest amount of captured data that answers your question. Start
with the default metadata-only mode; opt into content only when you need it, and
review the [security guidance](SECURITY.md#protecting-recordings) before sharing.

Metadata-only capture is the default: bodies are replaced with original types and
sizes without traversing builtin container contents. Exceptions retain only their
type; internal diagnostics never log exception bodies or tracebacks.

```python
from langchain_ollie import CapturePolicy, RecordingLimits, RecordingSession

with RecordingSession(
    capture=CapturePolicy.bounded(max_string_length=200),
    limits=RecordingLimits(max_spans=1000, max_events=2000),
) as recording:
    pass  # invoke your agent with recording.handler
```

Content capture is explicit and may expose sensitive text. `RedactionPolicy`
redacts matching mapping fields but is not a universal secret detector for free
text. Exception message capture requires constructing a `LangGraphCallbackHandler`
with `capture_exception_messages=True` and an appropriate content policy; stack
capture is not supported.
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

Choose memory recording for a short session you will export immediately, or
SQLite when you need to reopen traces later. In either case, finish agent work
before closing the recording and keep resource ownership explicit.

### Durable local recording

SQLite preserves records for later inspection without a service. Try the
close/reopen example with a new database path:

```bash
uv run --no-sync python examples/local_sqlite.py agent.db sqlite-trace.json
```

```python
from langchain_ollie import RecordingSession, SQLiteTraceStore

store = SQLiteTraceStore("agent.db")  # new dedicated recording database
try:
    with RecordingSession(store=store) as recording:
        result = app.invoke(inputs, config={"callbacks": [recording.handler]})
finally:
    store.close()
```

Later, from another process, list the traces and use one returned ID in place of
`TRACE_ID` to inspect or export it:

```bash
python -m langchain_ollie list agent.db --limit 20 --offset 0
python -m langchain_ollie summary agent.db TRACE_ID
python -m langchain_ollie export agent.db TRACE_ID trace.json
```

CLI inspection uses read-only connections. SQLite stores execution records rather
than prebuilt Perfetto output, so you can query them or export a timeline later.
Session loss diagnostics persist on orderly close. `TraceQuery` bounds listing reads;
`delete_traces(before_unix_ns=...)` removes whole finalized traces only. Exact
start/completion replays are idempotent; unknown schemas are never migrated
silently.

After a crashed writer, normal open preserves unfinished spans. **Only after all
previous writers have stopped**, opt into `SQLiteTraceStore(path, recover=True)`
or `python -m langchain_ollie recover agent.db`. Recovery marks work
abandoned and its duration as a last-observed lower bound. Do not reuse a nonempty
database with a new `RecordingSession`; reopen it for query/export or start a new
recording file. Prefer memory capture when minimizing callback I/O overhead.

### Session resource ownership

The context manager handles shutdown for the usual recording path. When supplying
your own store, keep its lifetime separate from the session's:

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

Use the tested modern runtime versions below. If you are upgrading from the
prototype, migrate to explicit session/callback attachment instead of trying to
preserve the old import paths.

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

Before submitting a code change, run the configured checks below. Default tests
use local scripted models, so provider credentials and network access are not
needed. See the [contributor guide](AGENTS.md) for architecture and test contracts.

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

Local tracing works without any of these additions. Enable a resource sampler
when process-level counters answer your question, or an export bridge when you
already have a destination for telemetry.

### Optional process resource counters

Add counters when you need to distinguish elapsed waits from process-wide CPU or
memory activity. They complement operation durations; they do not attribute
resources to individual overlapping agents.

From a checkout, install `pip install '.[resources]'` or
`uv sync --group dev --group agents --group resources`, then run:

```bash
uv run --no-sync python examples/local_resources.py resources.json
```

`from langchain_ollie.resources import ProcessSampler` is opt-in; no
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
See [sampler overhead and reproduction](benchmarks/REF09_RESOURCES.md) when choosing
an interval and sample cap.

### Optional OpenTelemetry export-only bridge

Use this bridge to send completed local spans to an existing SDK exporter while
keeping local recording independent of destination failures. It does not turn
the library into a distributed tracing integration.

From a checkout, install `pip install '.[otel]'`. The no-network SDK demo:

```bash
uv sync --group dev --group agents --group resources --group otel
uv run --no-sync python examples/local_otel.py otel-local.json
uv run --no-sync pytest -m otel
```

Use `OpenTelemetryProcessor(exporter)` from
`langchain_ollie.exporters.otel` in `RecordingSession(processors=[...])`.
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
Flush and shutdown bound the caller's wait but cannot kill an uncooperative
foreign exporter. A destination failure does not erase local Perfetto records.
The optional suite uses only an isolated loopback
OTLP receiver, never an external service; the default suite excludes it.

### Scope limits and alternatives

This library focuses on bounded, single-process execution evidence. Automatic
cross-process tracing, full distributed propagation and stack timeline correlation
are not provided. [ADR-001](docs/adr/001-profiler-correlation.md) explains why
automatic py-spy/Speedscope alignment is unreliable. The library does not install
or attach a profiler. For broader monitoring, evaluate
[OpenInference](https://github.com/Arize-ai/openinference) with local Phoenix,
MLflow Tracing or self-hosted Langfuse; this project focuses on standalone local
artifacts.
