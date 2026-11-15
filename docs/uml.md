# Classes and runtime interactions

These UML views complement the [architecture overview](architecture.md). They use
Mermaid, which GitHub renders directly; other Markdown viewers need Mermaid
support to display the fenced blocks. Diagrams show selected methods and fields,
not exhaustive API signatures. Names match the implementation.

**Notation:** solid arrows show associations, hollow diamonds group records,
filled diamonds indicate owned components, dashed arrows show use dependencies,
and dashed hollow-triangle arrows mean structural protocol conformance. Python
`Protocol` implementations need not inherit from the protocol. `_Recorder` is a
private adapter-local contract, not an additional public API.

## Recording and extension protocols

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

`BaseCallbackHandler` comes from `langchain_core`. The same ordinary `on_*` hooks
serve synchronous and asynchronous agent execution; there is no separate async
tracing engine. The adapter depends only on `_Recorder`, so translation can be
tested with a fake recorder without involving a store or exporter.

`TraceStore` writes are synchronous and atomic, with idempotent exact span
start/completion replays. Event/metric appends must not be automatically retried
after an uncertain acknowledgement. `FlushableTraceStore` is an independent,
optional capability, not a base class that every store must implement.

`PerfettoExporter` is intentionally **not** a `SpanProcessor`: it transforms a
snapshot on demand. `OpenTelemetryProcessor` implements `SpanProcessor` and
bridges completed spans to an explicitly supplied SDK exporter. It is imported
from its optional module and does not install a global provider or propagator.

Source: [session](../src/langgraph_instrumentation/session.py),
[callback adapter](../src/langgraph_instrumentation/adapters/langgraph.py),
[recorder](../src/langgraph_instrumentation/recorder.py),
[store protocols](../src/langgraph_instrumentation/stores/base.py),
[exporters](../src/langgraph_instrumentation/exporters/).

## Payload projection and privacy

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

Framework normalization lives in
[adapters/projection.py](../src/langgraph_instrumentation/adapters/projection.py);
general capture, redaction, metadata and usage policies live in
[policies.py](../src/langgraph_instrumentation/policies.py). Normalization happens
within bounded projection rather than first copying an unbounded object graph.
Metadata-only capture describes payload types and sizes without traversing
builtin container contents. Content capture remains subject to safety budgets,
and field redaction is not a universal secret detector for free text.

These policies precede the recorder/store trust boundary. Direct low-level
recorder and store callers must supply trusted JSON-compatible attributes and
apply their own capture policy; a store does not inherit recorder admission
budgets. See [safety limits](../README.md#safety-and-recording-limits).

## Neutral trace model

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

All these [records](../src/langgraph_instrumentation/models.py) are frozen
dataclasses with immutable nested attributes. Completion creates a replacement
`Span`; it does not mutate a previously returned snapshot. Optional fields such
as a root's parent ID, an active span's duration and a trace-scoped metric's span
ID can be absent (types in the diagram are abbreviated).

`Trace` contains recording identity and resource metadata; `Span` describes one
operation. Unix timestamps support external representation, while monotonic
timestamps determine local elapsed duration. `ExecutionLane` is a logical
visualization lane, not necessarily an OS thread. Events belong to spans, and
metrics can be trace- or span-scoped.

Parent IDs express **containment**. `SpanLink` represents a separate, explicitly
evidenced relationship and can reference an endpoint outside the bundle. Neither
nesting nor timing proves a graph transition, data dependency or critical path.

## Recording interaction

The successful path below follows a single operation; real agents interleave
callbacks from many nested or concurrent operations. Projection and neutral-record
creation are abbreviated. Store writes and processor callbacks are separate
destinations, not one atomic transaction.

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
processors still receive accepted completed spans. Diagnostics disclose that loss;
the diagram does not imply guaranteed delivery. See
[lifecycle and failure isolation](architecture.md#session-lifecycle-and-failure-isolation).

## Evaluation classes

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

Source: [runner](../src/langgraph_instrumentation/evals/runner.py),
[execution protocols/adapters](../src/langgraph_instrumentation/evals/adapters.py),
[evaluators](../src/langgraph_instrumentation/evals/evaluators.py),
[evaluation models](../src/langgraph_instrumentation/evals/models.py).
Continue with the [artifact and CLI guide](evals.md) or the
[reviewed dataset-improvement workflow](agent-eval-workflow.md).
