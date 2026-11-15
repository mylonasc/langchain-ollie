# Classes and runtime interactions

Most applications need only `RecordingSession`: attach its callback, run the agent,
and export a trace. Understanding the classes underneath helps when you need to
change persistence, add an export destination, or evaluate agent behavior.

Start with [how data moves through the library](architecture.md#data-flow), then
follow the sections below from recording to evaluation. For a working integration,
use the [quickstart](../README.md#quickstart-trace-a-langgraph-agent).

**Reading the relationships:** arrows connect collaborators; diamonds group owned
components or related records. A dashed arrow with a hollow triangle means that a
class provides the methods required by a protocol. In Python, implementing those
methods is enough; inheriting from the protocol is not required.

## Recording and extension protocols

The session connects three different jobs: translating callbacks, managing span
lifecycles, and delivering records. Keeping these jobs separate lets you change
storage or add a destination without changing the agent integration.

```mermaid
classDiagram
    direction TB
    class RecordingSession {
        +handler: LangGraphCallbackHandler
        +recorder: TraceRecorder
        +snapshot() tuple
        +export(destination)
        +close(timeout) bool
    }
    class BaseCallbackHandler {
        <<external>>
    }
    class LangGraphCallbackHandler {
        +on_chain_start()
        +on_chain_end()
        +on_chat_model_start()
        +on_llm_end()
        +on_tool_start()
        +on_tool_end()
    }
    class _Recorder {
        <<Protocol>>
        +start_span()
        +add_event()
        +record_metric()
        +observe_token()
        +end_span()
    }
    class TraceRecorder {
        +start_span()
        +end_span()
        +expire_spans() int
        +force_flush(timeout) bool
        +close(timeout) bool
    }
    class TraceStore {
        <<Protocol>>
        +start_span(trace, span)
        +complete_span(span)
        +record_event(event)
        +record_metric(metric)
        +get_trace(trace_id) TraceBundle
        +list_traces(query) tuple
        +delete_trace(trace_id) bool
        +close(timeout) bool
    }
    class FlushableTraceStore {
        <<Protocol>>
        +force_flush(timeout) bool
    }
    class MemoryTraceStore
    class SQLiteTraceStore
    class SpanProcessor {
        <<Protocol>>
        +on_end(span)
        +force_flush(timeout) bool
        +shutdown(timeout) bool
    }
    class OpenTelemetryProcessor
    class PerfettoExporter {
        +to_dict(bundles) PerfettoArtifact
        +export(bundles, destination)
    }

    RecordingSession *-- LangGraphCallbackHandler : creates
    RecordingSession *-- TraceRecorder : creates
    RecordingSession --> TraceStore : owns default only
    RecordingSession ..> PerfettoExporter : exports snapshots
    BaseCallbackHandler <|-- LangGraphCallbackHandler
    LangGraphCallbackHandler --> _Recorder : translates callbacks
    TraceRecorder ..|> _Recorder
    TraceRecorder --> TraceStore : optional persistence
    TraceRecorder ..> FlushableTraceStore : optional flush capability
    MemoryTraceStore ..|> TraceStore
    SQLiteTraceStore ..|> TraceStore
    TraceRecorder --> "0..*" SpanProcessor : completed spans
    OpenTelemetryProcessor ..|> SpanProcessor
```

`LangGraphCallbackHandler` uses LangChain's `BaseCallbackHandler` hooks for both
synchronous and asynchronous execution. It translates framework callbacks into
recorder operations; `_Recorder` is the private contract for those operations,
not an interface applications need to import.

`TraceStore` writes are synchronous and atomic, with idempotent exact span
start/completion replays. Event/metric appends must not be automatically retried
after an uncertain acknowledgement. `FlushableTraceStore` is an independent,
optional capability, not a base class that every store must implement.

`PerfettoExporter` is intentionally **not** a `SpanProcessor`: it transforms a
snapshot on demand. `OpenTelemetryProcessor` implements `SpanProcessor` and
bridges completed spans to an explicitly supplied SDK exporter. It is imported
from its optional module and does not install a global provider or propagator.

Choose [SQLite recording](../README.md#durable-local-recording) when traces must
survive a process exit, or the [OTel bridge](../README.md#optional-opentelemetry-export-only-bridge)
when completed spans should also go to an existing telemetry destination. For a
custom implementation, follow the [extension contracts](architecture.md#dependency-direction).

## Payload projection and privacy

Agent inputs, outputs and tool arguments can contain secrets or large objects.
The callback handler therefore transforms them before recording. Capture policy
decides how much to retain; redaction policy removes matching fields. Usage
extraction separately preserves reported token counts without requiring body capture.

```mermaid
classDiagram
    direction LR
    class LangGraphCallbackHandler
    class CapturePolicy {
        +metadata() CapturePolicy
        +bounded() CapturePolicy
        +full() CapturePolicy
        +capture(value)
    }
    class RedactionPolicy {
        +should_redact(path, key, value) bool
    }
    class MetadataProjector {
        +project(metadata) dict
    }
    class UsageExtractor {
        +extract(response) TokenUsage
    }
    class TokenUsage
    class Sanitizer {
        +sanitize(value)
    }
    LangGraphCallbackHandler --> CapturePolicy : payload policy
    LangGraphCallbackHandler --> RedactionPolicy : redaction rules
    LangGraphCallbackHandler --> MetadataProjector : selected metadata
    LangGraphCallbackHandler --> UsageExtractor : reported usage
    CapturePolicy ..> Sanitizer : bounded traversal
    Sanitizer --> RedactionPolicy : checks fields
    MetadataProjector --> RedactionPolicy : checks fields
    UsageExtractor ..> TokenUsage : returns
```

Metadata-only capture describes payload types and sizes rather than retaining
their contents. When you opt into content capture, `Sanitizer` limits traversal
and applies redaction as it goes, rather than first copying an unbounded object
graph. Field redaction cannot detect every secret in free text; review the
[artifact-sharing guidance](../SECURITY.md#protecting-recordings) before sharing a trace.

These policies precede the recorder/store trust boundary. Direct low-level
recorder and store callers must supply trusted JSON-compatible attributes and
apply their own capture policy; a store does not inherit recorder admission
budgets. See [safety limits](../README.md#safety-and-recording-limits).

## Neutral trace model

The library keeps execution evidence independent of the agent framework and the
viewer. A `TraceBundle` is a snapshot of one trace, its operations and its metrics.
Use these records to inspect a recording programmatically or build a consumer
without parsing Perfetto output.

```mermaid
classDiagram
    direction TB
    class TraceBundle {
        +trace: Trace
        +spans: tuple
        +metrics: tuple
    }
    class Trace {
        +trace_id: TraceId
        +service_name: str
        +resource_attributes: Attributes
    }
    class Span {
        +trace_id: TraceId
        +span_id: SpanId
        +parent_span_id: SpanId
        +kind: SpanKind
        +status: SpanStatus
        +duration_ns: int
    }
    class SpanEvent {
        +name: str
        +time_monotonic_ns: int
        +attributes: Attributes
    }
    class SpanLink {
        +trace_id: TraceId
        +span_id: SpanId
        +relationship: str
    }
    class ExecutionLane {
        +lane_id: str
        +name: str
    }
    class MetricPoint {
        +trace_id: TraceId
        +span_id: SpanId
        +values: Mapping
    }
    TraceBundle o-- "1" Trace
    TraceBundle o-- "0..*" Span
    TraceBundle o-- "0..*" MetricPoint
    Span o-- "0..*" SpanEvent
    Span o-- "0..*" SpanLink
    Span --> "0..1" ExecutionLane
    Span --> "0..1" Span : parent ID / containment
    SpanLink ..> Span : target IDs / may be external
    MetricPoint ..> Span : optional span ID
```

Snapshots are immutable: completing a span does not change a snapshot you already
hold. Some values are naturally absent: a root has no parent ID, an active span
has no final duration, and a trace-scoped metric has no span ID.

`Trace` contains recording identity and resource metadata; `Span` describes one
operation. Unix timestamps support external representation, while monotonic
timestamps determine local elapsed duration. `ExecutionLane` is a logical
visualization lane, not necessarily an OS thread. Events belong to spans, and
metrics can be trace- or span-scoped.

Parent IDs express **containment**. `SpanLink` represents a separate, explicitly
evidenced relationship and can reference an endpoint outside the bundle. Neither
nesting nor timing proves a graph transition, data dependency or critical path.
See [interpreting the timeline](../README.md#view-locally) for how these records
appear in Perfetto, and [the model definitions](../src/langchain_ollie/models.py)
for their complete fields.

## Recording interaction

Recording follows the application's lifetime: open a session before invoking the
agent, consume the result or stream, then close and export. Following one operation
through that sequence explains why export can happen after the session closes.
Nested and concurrent operations use the same path with separate IDs.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Session as RecordingSession
    participant App as LangGraph / Deep Agents
    participant Handler as LangGraphCallbackHandler
    participant Recorder as TraceRecorder
    participant Store as TraceStore
    participant Processor as SpanProcessor (optional)
    participant Exporter as PerfettoExporter

    User->>Session: create / enter
    User->>App: invoke(inputs, callbacks=[session.handler])
    App->>Handler: on_*_start(run_id, parent_run_id, payload)
    Note over Handler: Classify, project and redact
    Handler->>Recorder: start_span(correlation_id, parent, safe attributes)
    Note over Recorder: Assign IDs, timestamps and lane, enforce budgets
    Recorder->>Store: start_span(trace, span)
    App->>Handler: on_*_end(result)
    Handler->>Recorder: record_metric(...) / end_span(...)
    Recorder->>Store: record_metric(...) / complete_span(span)
    opt Configured processors
        Recorder->>Processor: on_end(completed immutable span)
    end
    App-->>User: agent result
    User->>Session: close() / exit scope
    Session->>Recorder: close()
    Note over Recorder: Abandon unfinished work, flush and shut down dependencies
    Session->>Store: list_traces() / get_trace()
    Store-->>Session: immutable TraceBundle snapshots
    Note over Session,Store: Cache snapshots, close only session-owned store
    User->>Session: export(destination)
    Session->>Exporter: export(cached bundles, diagnostics, destination)
    Exporter-->>User: local Perfetto JSON artifact
```

Close is idempotent; snapshots/export remain available after a successful close.
In best-effort mode, store failure can leave a partial local recording while
processors still receive accepted completed spans. Store and processor delivery
are independent, so successful delivery to one does not guarantee the other.
Check session diagnostics for loss and see
[lifecycle and failure isolation](architecture.md#session-lifecycle-and-failure-isolation).

## Evaluation classes

Tracing explains what happened; evaluation asks whether it was acceptable.
`EvalRunner` combines an execution adapter with checks over answers and trace
evidence. This lets you test both correctness and execution costs, such as whether
a required tool ran or a model-call budget was exceeded.

```mermaid
classDiagram
    direction TB
    class EvalRunner {
        +run(cases) EvalRun
        +arun(cases) EvalRun
    }
    class ExecutionAdapter {
        <<Protocol>>
        +invoke(invocation) EvalOutput
    }
    class AsyncExecutionAdapter {
        <<Protocol>>
        +ainvoke(invocation) EvalOutput
    }
    class LangGraphEvalAdapter
    class EvalInvocation
    class RecordingSession
    class Evaluator {
        <<Protocol>>
        +name: str
        +evaluate(context) EvalAssessment
    }
    class ExactText
    class ToolCalled
    class MaxMetric
    class FunctionEvaluator
    class EvalContext
    class EvalAssessment

    EvalRunner --> ExecutionAdapter : sync execution
    EvalRunner --> AsyncExecutionAdapter : async execution
    LangGraphEvalAdapter ..|> ExecutionAdapter
    LangGraphEvalAdapter ..|> AsyncExecutionAdapter
    EvalRunner ..> RecordingSession : fresh per case trial
    EvalRunner ..> EvalInvocation : creates per turn
    EvalInvocation --> RecordingSession : active recording
    LangGraphEvalAdapter ..> EvalInvocation : attaches session.handler
    EvalRunner ..> EvalContext : builds from results and traces
    EvalRunner --> Evaluator : runs checks
    ExactText ..|> Evaluator
    ToolCalled ..|> Evaluator
    MaxMetric ..|> Evaluator
    FunctionEvaluator ..|> Evaluator
    Evaluator ..> EvalContext : consumes
    Evaluator ..> EvalAssessment : returns
```

The two adapter protocols separate sync and async capabilities. The built-in
`LangGraphEvalAdapter` implements both, provided the wrapped application supports
the requested invocation method. A custom execution adapter uses the supplied
session rather than creating a second recorder.

For each case trial, the runner invokes turns in sequence with isolated history
or a trial-specific checkpoint thread ID. After each turn it measures new trace
bundles and evaluates turn checks. Conversation checks run after completed
execution. `EvalContext` supplies output, metrics, traces and prior turn results;
an `Evaluator` returns an `EvalAssessment`. The diagram shows representative
checks; text, state, metric and optional judge checks share the same protocol.

### Definitions and results

Keep the intended behavior separate from the observed behavior. Cases and turns
define the test; runs, trials and turn results record what happened. A trial is
one execution of a case, so repeated trials reveal inconsistent behavior without
changing the case definition.

```mermaid
classDiagram
    direction TB
    class EvalCase
    class EvalTurn
    class Evaluator {
        <<Protocol>>
    }
    class EvalRun
    class EvalTrial
    class EvalTurnResult {
        +trace_ids: tuple
        +execution_status: EvalExecutionStatus
    }
    class EvalOutput
    class EvalMetrics
    class EvalAssessment

    EvalCase o-- "1..*" EvalTurn
    EvalCase --> "0..*" Evaluator : conversation checks
    EvalTurn --> "0..*" Evaluator : turn checks
    EvalRun o-- "0..*" EvalTrial
    EvalTrial o-- "0..*" EvalTurnResult
    EvalTrial o-- "0..*" EvalAssessment : conversation assessments
    EvalTurnResult o-- "0..*" EvalAssessment : turn assessments
    EvalTurnResult --> "0..1" EvalOutput
    EvalTurnResult --> "1" EvalMetrics
```

Definitions (`EvalCase`, `EvalTurn`), observations (`EvalOutput`, turn/trial
results), measurements (`EvalMetrics`) and judgments (`EvalAssessment`) remain
separate. Execution failure is not an ordinary failed check. Results retain trace
IDs, not a duplicate timeline; trace-derived metrics record provenance and
completeness. Aggregation and comparison operate on those results afterward.

Try [your first evaluation](evals.md#run-your-first-evaluation), then use the
[artifact and CLI guide](evals.md#save-and-reopen-results) or the
[reviewed dataset-improvement workflow](agent-eval-workflow.md).
