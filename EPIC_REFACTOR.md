# EPIC: Local-First Agent Performance Tracing

## Status and scope

- Status: REF-01 through REF-11 complete; `epic-refactor` was fast-forward merged
  into `main` at `67d05ee` on 2026-11-14 (REF-10 is an evidenced no-go, not a
  profiler implementation). All eleven post-merge hosted jobs passed:
  https://github.com/mylonasc/langgraph_instrumentation/actions/runs/43583712207.
- Parent issue: [#13](https://github.com/mylonasc/langgraph_instrumentation/issues/13).
- Primary outcome: safe, bounded, offline LangGraph and Deep Agents execution
  traces that can be inspected in Perfetto without a service or credentials.
- Delivery: small vertical slices, clean API breaks, no legacy compatibility work.
- Core completion: REF-01 through REF-08. Optional follow-ups: REF-09 through REF-11.
- This plan supersedes the FRK roadmap and architectural commitments in the
  previous version of this file, including making OTLP a release prerequisite.
   Git history preserves that plan. The historical issues were reconciled after
   merge: #7–#9 and #11 closed as delivered; #1 and #10 closed as superseded, not
   literal completion of their original requirements. #12 now indexes focused
   follow-ups. See [the issue review](docs/ISSUE_REVIEW.md).
- Preserve unrelated uncommitted work; the merge and issue review did not touch
  the user's untracked `main.py`.

## Product decisions

1. **Perfetto first.** Optimize for local latency investigation, parallel agent
   execution, portable artifacts, and low instrumentation overhead. This is not
   initially a hosted observability service or a Python CPU profiler.
2. **Modern agents only.** Target a deliberately narrow, tested modern Python,
   LangGraph, and `deepagents` version matrix. Keep `langchain-core` callbacks and
   message types only where required by those runtimes. Do not promise standalone
   legacy LangChain chains, old callback signatures, old import paths, or provider
   response formats without current agent e2e coverage. Drop uncertain features
   instead of accumulating compatibility branches.
3. **Rewrite when simpler.** Reuse useful models/tests, not prototype behavior.
   Delete the legacy instrumentation, logger, async-hook shims, hashed task/tool
   tracks, and compatibility exports. No requirement to preserve old JSON bytes
   or unsafe payload capture. Do not delete the only working Perfetto route until
   its replacement is demonstrated in REF-02.
4. **Safe defaults.** Metadata-only payload capture; exception type but no message
   or stack by default. Capture never invokes arbitrary `str`/`repr`. No credentials,
   prompt bodies, tool bodies, or exception bodies in internal diagnostics.
5. **Bounded by default.** Enforce finite recording budgets from the first usable
   release. Explicitly report dropped/truncated data; never silently claim a
   complete trace. Full payload capture is explicit, redacted, and still subject
   to overall recording safety limits.
6. **OTel is an optional bridge, not a replacement for Perfetto.** Do not make the
   early local workflow depend on a collector or rebuild a general distributed
   tracing SDK. Prefer supported OTel APIs/SDK machinery when REF-11 is reached.
7. **Measurement honesty.** Callback wall duration is not CPU time. Process RSS
   is not per-agent allocation. Sampling cannot establish exact attribution.
   Record measurement source, units, limitations, and dropped samples.

## Baseline and review findings

The working tree already has packaging, neutral immutable models, policies,
recorder, memory store, and an in-progress modern callback adapter. At review,
141 tests passed (85% aggregate coverage), lint and type checks passed, one test
file needed formatting, and there was no CI configuration. The modular Perfetto
exporter, SQLite store, and OTLP implementation were absent.

The execution slices below address these observed gaps:

- Exception messages bypass capture/redaction even with capture disabled.
- Handler-wide message deduplication produces incomplete independent traces.
- Framework preprocessing truncates full capture and corrupts original sizes.
- Trace-count limits do not bound a large/unfinished trace; lane/context caches
  can retain data beyond its useful recording lifetime.
- Event appends repeatedly copy/validate/sort full event collections.
- Store failures cascade into lost parent resolution and suppress processors.
- Parentage is confused with graph causality; node names are not execution tracks.
- Processor inputs lack trace resource context; shutdown/ownership are complex.
- The documented quickstart still uses the less-safe legacy implementation.

## Target boundaries

```text
Modern LangGraph / Deep Agents (real runtimes, explicit callback attachment)
                              |
                   thin framework translation
                              |
                  capture + bounded local recorder
                              |
             neutral versioned recording / snapshots
                      /                  \
              memory (first)         SQLite (later)
                      \                  /
                       Perfetto exporter
                              |
               JSON artifact + local summary/CLI

Optional later consumers: process-resource sampler, profiling import,
                          OTel bridge using standard SDK components
```

- Models and stores do not import frameworks or exporters. Framework-specific
  projection belongs at the adapter boundary, not in neutral policies.
- Keep mutable bounded buffers internally; freeze records at public snapshot
  and completion boundaries. Avoid rebuilding all accumulated events on append.
- Correlation context contains IDs/minimal lifecycle data, never root payloads.
- Resource/service identity accompanies completed records destined for consumers;
  exporters must not query concrete stores to reconstruct missing context.
- Separate containment parents, causal links, runtime execution identity, and
  visual track allocation. Do not fabricate causality unavailable from the runtime.
- Use monotonic time for local durations and wall time for interoperability;
  identify clock/process domains and avoid comparing unrelated monotonic clocks.
- No plugin registry, PostgreSQL adapter, remote query service, or generalized
  processor ecosystem is required for the local product.

## Indexed phases and dependencies

Dependencies are completion gates, not suggestions. Work can be prepared in
parallel, but a slice is not accepted before all its prerequisites pass. IDs are
stable and may become child issues later; this parent initially tracks the slices.

| Phase | Slice | Deliverable | Explicit prerequisites | Priority |
| --- | --- | --- | --- | --- |
| A | REF-01 | Safety, supported scope, regression harness, baseline CI | None | P0 |
| A | REF-02 | First complete modern LangGraph-to-Perfetto workflow; remove legacy | REF-01 | P0 |
| B | REF-03 | Real Deep Agents offline e2e and runnable demo | REF-02 | P0 |
| B | REF-04 | Bounded, efficient recording under load | REF-02 | P0 |
| C | REF-05 | Accurate graph, concurrency, retry and resume semantics | REF-03, REF-04 | P1 |
| C | REF-06 | Failure isolation and lifecycle simplification | REF-04 | P1 |
| D | REF-07 | Durable SQLite recording and artifact reopen | REF-05, REF-06 | P1 |
| D | REF-08 | Local-first release and performance acceptance gate | REF-03, REF-04, REF-05, REF-06, REF-07 | P1 |
| E | REF-09 | Optional CPU/RSS process sampling in Perfetto | REF-08 | P2 |
| E | REF-10 | Optional sampling-profiler interoperability | REF-09 | P3 |
| E | REF-11 | Optional OpenTelemetry interoperability | REF-08 | P3 |

### Execution evidence

- REF-01: [#14](https://github.com/mylonasc/langgraph_instrumentation/issues/14).
  Implemented capture/exception privacy, stateless framework projection, finite
  lifetime admission/payload budgets with loss snapshots, current version bounds,
  offline guards and CI. Reviewed and incorporated the pre-existing adapter/policy
  work; unrelated `main.py` and ignored generated artifacts remain untouched.
  Verification: 159 tests pass on standard CPython 3.12, 3.13 and 3.14; lint,
  formatting, types, sdist/wheel build and installed-wheel recording smoke pass.
  Hosted CI/current-resolution jobs run when the branch is pushed; this is not
  a claim of completed Deep Agents e2e (REF-03) or pressure-aware retention (REF-04).
- REF-02: [#15](https://github.com/mylonasc/langgraph_instrumentation/issues/15).
  M1 delivered: `RecordingSession`, immutable snapshots/export after shutdown,
  explicit store ownership, atomic deterministic Perfetto output, non-overlapping
  tracks, numeric containment flow IDs, process/clock/schema identity, and loss
  diagnostics including retention/destination failures. Removed prototype code,
  compatibility-only tests, classic completion/retriever hooks, provider-dependent
  demos, unused extras, and lint/type exclusions. Added a scripted tool-calling
  LangGraph example and a separate required checksum-pinned Perfetto CI job.
  Verification: 166 default tests pass on Python 3.12/3.13/3.14; two real sync/async
  agent artifacts pass Perfetto v59.1 SQL checks for span durations/IDs/parents,
  token counters, containment flows, and zero importer errors. Lint, formatting,
  types, package build, core-only wheel smoke, and installed-wheel agent demo pass.
  Example: `uv run --no-sync python examples/local_agent.py trace.json`.
  Hosted CI is not yet claimed; the implementation commits remain local.
- REF-03: [#16](https://github.com/mylonasc/langgraph_instrumentation/issues/16).
  Added a real offline Deep Agent using state-backed write/read tools and explicit
  delegation to a scripted worker with a local multiplication tool. Shared model
  support implements finite scripts, tool binding, sync/async generation,
  fragmented tool-call streaming, copied responses, and once-per-call usage.
  Required tests cover invoke/ainvoke/stream/astream, nested worker containment,
  file/output correctness, tool-call IDs, privacy/truncation and repeated isolated
  invocations in one session. Deep Agents roots are now correctly classified as
  graphs from their observed `ls_integration="deepagents"` metadata.
  Verification: 186 default tests pass on Python 3.12/3.13/3.14, plus six actual
  Perfetto importer tests (including all four Deep Agents execution APIs). Lint,
  format, types and package build pass. A clean installed-wheel Deep Agents demo
  and full default/importer suites also pass with current dependencies (LangGraph
  1.2.22 versus locked 1.2.21). No provider credentials, network or shell tools are
  used by the scenarios. CI now smoke-tests both installed-wheel examples.
  Example: `uv run --no-sync python examples/local_deep_agent.py deep-agent-trace.json`.
  This slice's initial evidence preceded REF-04; M2 is now complete below.
- REF-04: [#17](https://github.com/mylonasc/langgraph_instrumentation/issues/17).
  Added aggregate/per-trace encoded attribute budgets, per-trace record caps,
  active/per-span event caps, cooperative expiration, payload-free streaming chunk
  summaries and capped opt-in detail. Memory events append without rebuilding
  spans; snapshots/completion freeze and sort once. Stress tests prove bounded
  retention, expiry and immutable snapshot behavior. Benchmark evidence and review
  thresholds are in `benchmarks/REF04_BASELINE.md` (event doubling 2.02x; about
  0.43 ms metadata overhead per minimal graph in this local baseline).
  Verification: 192 default tests and six Perfetto tests pass; lint/format/types
  and wheel/sdist build pass. Prior slices are pushed; hosted CI for REF-03 passed
  at https://github.com/mylonasc/langgraph_instrumentation/actions/runs/43575876930.

- REF-05: [#18](https://github.com/mylonasc/langgraph_instrumentation/issues/18).
  Added bounded immutable explicit span links and Perfetto causal arrows distinct
  from containment; checkpoint resume is a new linked invocation. Unknown parents
  become explicit partial roots in default mode. Root outcome now wins over
  handled retry failures, and runtime interrupts are marked as neutral control
  flow. Real tests cover Send fan-out with barriers, nested repeated nodes/loops,
  retries/terminal failures, checkpoint resume, shared-handler concurrency and
  concurrent Deep Agents workers. Imported SQL verifies links and overlap-free
  tracks. Verification: 201 default tests, seven importer tests, lint/format/types
  and build pass. Runtime metadata is preserved; unsupported automatic DAG-edge
  inference is deliberately not fabricated.

- REF-06: [#19](https://github.com/mylonasc/langgraph_instrumentation/issues/19).
  Best-effort lifecycle now survives destination rejection and dispatches completed
  resource-bearing spans to independent processors. Strict mode retains explicit
  retry semantics. Queue admission is bounded, session finalization is serialized
  without locking processor execution, and exact store start/completion replays
  are idempotent. Documented ambiguous append acknowledgements and cooperative
  timeout limits. Real tests inject all store phases/export failure, slow stores,
  concurrent close, pending-pressure and reentry, early Deep Agents stream closure
  and cancellation inside an async tool. Verification: 212 default tests, seven
  importer tests and lint/format/types/build pass; no application-error replacement
  or payload-bearing failure logs. M3 is delivered with REF-05.

- REF-07: [#20](https://github.com/mylonasc/langgraph_instrumentation/issues/20).
  Transactional SQLite schema v1 stores neutral records, links, resources and loss
  diagnostics with WAL/FULL sync, bounded listing, retention and read-only CLI.
  Explicit stopped-writer recovery marks abandoned lower-bound durations. Shared
  store contracts, process exit/reopen, disk-full/locked/rollback/schema mismatch,
  cascade cleanup and diagnostic-failure tests pass. Real sync/async LangGraph and
  Deep Agents recordings reopen identically and import into Perfetto.
  Verification: 232 default tests, 11 importer tests, lint/format/types and build.

- REF-08 release-gate implementation: [#21](https://github.com/mylonasc/langgraph_instrumentation/issues/21).
  Added minimum/locked/current required CI, socket-blocked clean-wheel quickstarts
  including SQLite/CLI, MIT metadata, security/upgrade/schema policy and `py.typed`.
  Reviewed scoped guides and removed stale progress claims. Local evidence: 232
  default tests on CPython 3.12/3.13/3.14, 11 importer tests, lint/format/types/build,
  and clean-wheel offline demos outside checkout pass. All ten required hosted jobs
  passed: https://github.com/mylonasc/langgraph_instrumentation/actions/runs/43579733564.
  REF-07's current-toolchain annotation warning is fixed. M4/core acceptance is met;
  this is not a package publication or merge authorization.

- REF-09: [#22](https://github.com/mylonasc/langgraph_instrumentation/issues/22).
  Opt-in psutil extra and bounded ProcessSampler add process-scope CPU/RSS counters,
  explicit one-core normalization, units/PID, overhead and missing/drop diagnostics.
  No default sampler import/thread/dependency loading. Deterministic sources,
  permission failures, admission drops, real subprocess work, cleanup and imported
  counter tracks pass. Local overhead: 48.8 µs median/sample (REF09_RESOURCES.md).
  Verification: 240 default tests, 12 importer tests, lint/format/types/build.

- REF-10: [#23](https://github.com/mylonasc/langgraph_instrumentation/issues/23).
  Completed compatibility spike with a documented **no-go**, not a shipped profiler.
  py-spy Speedscope count weights lack per-sample clock/process anchors and discard
  gaps; automated ambiguity fixtures prove automatic elapsed-span correlation is
  not recoverable. ADR `docs/adr/001-profiler-correlation.md` records source evidence,
  platform/privacy/permission limits, external opt-in commands and revisit criteria.
  No process inspection or elevated permissions were used; live-profiler overhead
  is explicitly unmeasured, not advertised. No fragile fallback engine added.

- REF-11: [#24](https://github.com/mylonasc/langgraph_instrumentation/issues/24).
  ADR-002 chooses an explicit export-only neutral consumer over an SDK callback
  adapter. Optional SDK conversion preserves IDs/parents/links/resources/events
  and timing, retaining local kind/status metadata without claiming unsupported
  distributed context or GenAI/OpenInference conventions. SDK batching/OTLP handle
  delivery; shared flush/shutdown workers bound caller waits even when SDK 1.45.2
  ignores its flush timeout. Rejections now increment recorder failure diagnostics
  as well as returning false. SDK failure/isolation/timeout/context tests and real
  offline-agent OTLP/protobuf transport to an isolated loopback collector pass.
  Verification: 242 default tests, 12 Perfetto tests, seven optional OTel tests,
  lint/format/types/build and no-network SDK demo. Optional tests have a required
  dedicated CI job and are excluded from the default core suite. REF-09/10 hosted
  CI also passed: actions/runs/43580038439 and actions/runs/43580262117.
- Final contract review: memory now preserves the original immutable start record
  like SQLite, so conflicting attribute replays after completion cannot masquerade
  as exact idempotent starts. A shared regression covers both implementations.
  Final local verification: 244 default tests, 12 importer tests and seven optional
  OTel tests; lint/format/types/build pass. Unrelated `main.py` remains untouched.

### Useful-state checkpoints

- **M1 / REF-02:** install, run a no-network modern LangGraph agent, open a safe
  bounded Perfetto trace. Stop here if this meets the user's needs.
- **M2 / REF-03 + REF-04:** run a real Deep Agent with tools/subagent delegation;
  capture it locally with tested budgets and an overhead report.
- **M3 / REF-05 + REF-06:** reliable concurrent/retrying/resumable agent diagnostics,
  explicit partial traces, and predictable lifecycle under failures.
- **M4 / REF-08:** supported local-first release with durable recording and
  reproducible installation, docs, artifact validation, and measured overhead.
- **M5 / optional:** resource counters, profiling correlation, or OTel can ship
  independently after M4. None blocks the useful local product.

## Execution slices

### REF-01 — Safety, scope, and regression foundation

**Depends on:** none. **Outcome:** trustworthy capture primitives and an executable
modern-agent baseline without claiming a finished new tracing workflow.

Implementation:

- Inventory tracked code/artifacts and in-progress changes. Pick exact tested
  framework versions plus a minimum/current matrix; lock the development baseline.
  Deep Agents belongs in the agent-test/example dependency set, not neutral core.
- Add regressions for capture-disabled/default exception leakage, independent
  traces sharing messages, failed writes before message capture, original metadata
  sizes, full capture, hostile values, and large/deep/cyclic structures.
- Gate exception messages/stacks separately; use one policy-aware normalization
  pass. Summarize originals without projecting bodies in metadata mode.
- Remove cross-trace message deduplication rather than inventing a blob store.
  Clarify full mode as content opt-in with explicit global safety truncation.
- Remove payloads from correlation caches and avoid duplicating root input in
  trace-level metadata. Define trusted direct-recorder inputs versus adapter policy.
- Establish finite baseline record/event/payload budgets and visible incomplete
  diagnostics; REF-04 expands pressure handling, not permission to ship unbounded.
- Fix formatting and add CI for lint, formatting, types, tests, build, and clean
  wheel import. Update `AGENT.md` and scoped guides to REF IDs and narrow scope.
- Mark the legacy API unsafe/deprecated in documentation during this short stage;
  do not add features or compatibility fixes to it. Cross-link prior FRK issues to
  this parent without automatically closing or rewriting their history.

Acceptance:

- Privacy canaries are absent from records and captured internal logs in none/
  metadata modes, including exceptions, tool failures, and instrumentation failures.
- Independent recordings are self-contained; unsupported objects are never formatted.
- Regression suite and all configured quality gates pass from a clean environment.
- Public limits and supported/dropped feature lists are documented and tested.

### REF-02 — Ship the first Perfetto vertical slice; delete legacy

**Depends on:** REF-01. **Outcome:** M1, usable before durable storage or OTel exists.

Implementation:

- Provide a small explicit session/context API over memory recording, callback
  attachment, snapshot/export, and close. Define ownership of injected stores;
  export-before-close must not be a hidden lifecycle requirement.
- Finish/rewrite only modern graph/node/chat-model/tool translation needed by
  the real agent fixture. Do not preserve uncertain classic LLM/retriever callbacks.
- Implement Perfetto complete spans, instant events, usage counters, process/track
  metadata, and clearly labeled containment flows. Allocate collision-free tracks
  for overlapping intervals; use logical node labels, not Python callback threads.
- Add schema version, resource/clock identity, loss summary, and atomic path writes.
  Support compact deterministic output and file-like output without `default=str`.
- Add a no-network tool-calling agent and public-API export example; document
  viewing locally, including a self-hosted/offline viewer option for sensitive data.
- Validate JSON via pinned Perfetto Trace Processor ingestion and SQL assertions
  in a required artifact CI job. Provision the binary during CI setup, never by an
  implicit test-time download. Keep one manual UI screenshot/check as supplementary.
- Delete the prototype module/exports/logger/shims and their compatibility-only
  tests after the new e2e passes. Rewrite examples/imports; remove stale generated
  files only when tracked, obsolete, and not user work. Remove tooling exclusions.

Acceptance:

- A real synchronous and asynchronous LangGraph agent calls a scripted model and
  local tool, returns the expected result, and emits a valid inspectable trace.
- Imported slices/parents/tracks/durations/counters match runtime semantics; spans
  remain correct when start/end callbacks occur on different threads.
- Re-exporting an unchanged snapshot is deterministic. Interrupted path export
  preserves the previous destination. Budget loss is visible in the artifact.
- Installed-wheel example works outside the checkout with no provider, service,
  OTel SDK, or network. No legacy public entry points remain.

### REF-03 — Real Deep Agents end-to-end coverage

**Depends on:** REF-02. **Outcome:** M2 agent functionality, not mocked callbacks.

Implementation:

- Add a tool-capable scripted chat model implementing the current runtime model
  contract, including `bind_tools`, sync/async generation, streamed chunks, tool
  call IDs, and usage. Script responses deterministically and fail on extra calls.
- Construct actual `deepagents.create_deep_agent` agents with that model, local
  tools, and an in-memory or temporary-directory backend restricted to the fixture.
  Configure subagent models explicitly; do not allow provider-default fallbacks.
- Exercise a custom tool, supported built-in file read/write tools, and delegation
  to a configured subagent. Add a stream-consumption scenario and repeated runs
  using one instrumentation session. Adapt to the pinned API rather than mock it.
- Add a documented Deep Agents quickstart that exports Perfetto through public
  APIs. Additional framework callbacks are supported only if fixtures require them.

Acceptance:

- Actual agent output/file content proves tools and delegation executed, while
  exported/imported traces contain their model/tool/subagent relationships.
- Sync/async semantic assertions pass without requiring identical callback counts
  or unstable middleware names. Usage is attributed once per actual model call.
- None/metadata canaries are absent from artifact, store snapshots, and logs;
  bounded mode retains only permitted content with explicit truncation.
- No network, environment credentials, shell execution, paid APIs, or external
  tracing backend is needed; CI runs these tests as required, not skipped extras.

### REF-04 — Bounded recording and efficient hot paths

**Depends on:** REF-02. **Outcome:** bounded long-running local diagnostics.

Implementation:

- Enforce global/per-trace byte and record budgets, active-span limits, event/
  token limits, retention, lane/cache limits, and unfinished-span expiration.
  Specify admission/drop/eviction policy and account for retained context too.
- Use append-oriented mutable internal buffers; validate/freeze/sort at snapshot
  boundaries. Default streaming to first-token/last-token/count/usage summaries,
  with explicitly capped token-event opt-in; do not redact a semantic token field
  accidentally as if it were an authentication credential.
- Keep minimal structural lifecycle state when payload/event capture is dropped.
  Every accepted snapshot stays internally consistent or explicitly incomplete.
- Add benchmarks for no-op versus instrumentation, event-count scaling, payload
  policies, concurrent graphs, and many small traces versus one large trace.

Acceptance:

- Stress tests demonstrate bounded retained records/bytes and reclaimable state
  after completion/expiry/eviction; use measured memory checks with tolerances,
  not exact RSS expectations. One endless trace cannot bypass the configured caps.
- Event append work no longer repeatedly sorts/copies the accumulated stream.
- Loss counters/reasons survive export. Benchmark method and results are committed;
  budgets/regression thresholds are explicit before this slice is closed.

### REF-05 — Graph causality, execution identity, and resume

**Depends on:** REF-03, REF-04. **Outcome:** accurate visualization of real agents.

Implementation:

- Separate containment from typed causal links; record execution-instance IDs,
  subgraph namespaces, graph steps, retry attempts, checkpoint/session identity,
  and resume relationships using supported runtime metadata.
- Render concurrent instances of the same node and identically named subgraphs
  on non-conflicting tracks. Add causal arrows only where runtime evidence exists.
- Define session versus invocation versus trace. Prefer separate resumed-invocation
  traces linked by checkpoint/session over pretending a process-local span survived
  restart. Document unsupported cross-process causality rather than guessing it.
- Distinguish handled/retried attempt errors from final root outcome. Preserve
  failed attempts without falsely reporting every recovered invocation as failed.

Acceptance:

- Real LangGraph e2e covers static fan-out/fan-in, `Send` dynamic fan-out, nested
  subgraphs, repeated node names, loops, retry-then-success, terminal failure,
  interrupt/checkpoint/resume, and concurrent invocations sharing a handler.
- Deep Agents delegation/concurrent subagent scenarios retain correct containment
  and stable invocation isolation. Perfetto-import assertions verify track layout.
- Unknown parents/late children yield an explicit partial trace or bounded documented
  recovery; they do not silently disappear or attach to an unrelated invocation.

### REF-06 — Failure isolation and lifecycle simplification

**Depends on:** REF-04. **Outcome:** recording cannot derail agent execution.

Implementation:

- Simplify recorder synchronization and shutdown around measured requirements.
  Keep explicit strict test mode; preserve the agent's original error/cancellation.
- Separate recording admission from destination success: a persistence/export
  failure must not erase correlation state or prevent independent consumers from
  receiving already accepted data. Local failure is visible, bounded, and nonfatal
  by default; do not promise durability when the authoritative store has failed.
- Specify idempotent mutations/retries and partial-write semantics. Do not promise
  that every exception from a future remote store proves its transaction rolled back.
- Specify timeout units/budgets, cooperative versus caller wait bounds, thread
  ownership, close retry behavior, flush consistency, reentry rules, and store
  ownership. No locks while calling user processors/exporters.

Acceptance:

- E2e failure injection covers start/event/complete/export failure, slow destination,
  cancellation during invoke/stream, early stream close, repeated/concurrent close,
  and shutdown with unfinished work. Deadlock tests have bounded watchdogs.
- Original results/errors are preserved by default; diagnostics disclose partial
  traces without payloads or unbounded warning storms. No worker leaks remain.
- Unit tests use events/barriers/fake clocks, not sleeps to assert callback order.

### REF-07 — Durable local SQLite recording

**Depends on:** REF-05, REF-06. **Outcome:** recordings survive process restart.

Implementation:

- Add versioned schema, transactional append/completion, indexes, safe connection
  ownership, WAL/busy policy, retention, and clear disk-full/locked/corrupt errors.
- Define crash recovery for unfinished spans and persisted loss/resource/context
  metadata; persist links and recording schema, not exporter-specific dictionaries.
- Share store contract tests with memory. Add local CLI/API list, summary, and
  export of a reopened trace, with pagination and bounded reads where needed.

Acceptance:

- LangGraph and Deep Agents scenarios run against both stores and export equivalent
  semantic Perfetto artifacts after SQLite close/reopen.
- Subprocess kill/reopen, migration/version mismatch, transaction rollback,
  concurrent access, retention, disk/write failure, and idempotent retry are tested.
- Recovery distinguishes abandoned/partial work from successful completion;
  deleting a trace removes its events, metrics, and links without dangling records.

### REF-08 — Local-first release gate

**Depends on:** REF-03, REF-04, REF-05, REF-06, REF-07. **Outcome:** M4 release.

Implementation and acceptance:

- Run the e2e matrix below in required CI jobs; install clean built wheels outside
  the source tree, test minimum/current supported dependencies and Python versions,
  and ensure narrow framework bounds match the tested combinations.
- Publish runnable no-network graph/deep-agent demos, memory/SQLite export, local
  viewing, loss/privacy semantics, lifecycle ownership, upgrade notes, and known
  blind spots. Document that node wall duration includes callback overhead and is
  not CPU execution time or a guaranteed critical-path analysis.
- Ship license/security/release metadata, `py.typed`, schema/version policy, and
  benchmark reproduction commands. Test every documented quickstart.
- Include benchmark evidence for overhead/scaling and agreed regression budgets;
  do not make noisy shared-runner timing an unexplained flaky correctness gate.
- No legacy module/tool exclusions, compatibility-only tests, stale examples, or
  contradictory agent guides remain. Review artifacts for secret retention.
- All core slices pass acceptance. Optional OTel/profiling absence does not block
  release. No new framework compatibility claim without a required real-agent test.

### REF-09 — Optional process CPU and memory counters

**Depends on:** REF-08. **Priority:** lower; separately installable/opt-in.

- Sample process CPU time/utilization and RSS at configurable bounded intervals,
  optionally via `psutil`. Record timestamp, PID/process identity, units, CPU
  normalization, sampler overhead, interval, and missing/dropped samples.
- Render time-aligned Perfetto counters. Do not attribute process-wide CPU/RSS to
  overlapping agents; child-process tracking requires explicit separate identity.
- Test fake samplers deterministically; add subprocess smoke tests for CPU/memory
  work with broad tolerances, permission errors, disablement, and sampler cleanup.
- Acceptance: useful counters with no unsupported attribution, bounded buffering,
  documented overhead, and no sampler/dependency loaded in default installations.

### REF-10 — Optional sampling-profiler correlation

**Depends on:** REF-09. **Priority:** exploratory, lower than resource counters.

- Time-box a compatibility spike with an existing maintained profiler (for example
  py-spy or a supported Python sampling API); prefer importing/correlating existing
  output over writing a sampling engine. Document platform/Python/permission limits.
- Align a supported profile format with recording clock/process domains and expose
  stack samples alongside Perfetto execution tracks or a clearly linked companion
  artifact. Do not claim unsupported direct Perfetto import or exact task attribution.
- Apply path/stack privacy policy and sample/size caps. No production attach,
  elevated privileges, or subprocess inspection unless explicitly requested.
- Acceptance: reproducible opt-in example, automated fixture correlation tests,
  overhead measurements, unsupported-platform behavior, and a go/no-go ADR. If the
  spike fails, close with evidence and documented external profiling instructions
  rather than adding a fragile bespoke profiler.

### REF-11 — Optional OpenTelemetry monitoring bridge

**Depends on:** REF-08. **Priority:** lower; independent of REF-09/REF-10.

- Write an ADR comparing an OTel SDK-backed adapter/consumer with neutral-record
  conversion. Reuse standard batching/export/retry facilities rather than building
  a competing SDK. Keep the local path independent of OTel installation/config.
- Carry resource/service/schema context with completed records; define incoming
  parent context, trace flags/tracestate, links, sampling, and outgoing/manual child
  propagation. Explicitly distinguish full context interoperability from export-only.
- Map versioned GenAI/OpenInference conventions deliberately; preserve richer local
  kinds/statuses as namespaced attributes rather than conflating OTel span kinds.
- Acceptance: in-memory SDK tests plus local collector e2e prove parentage/resource
  mapping, optional context propagation, bounded flush, destination isolation, and
  unchanged Perfetto behavior. No external collector in the default core suite.

## End-to-end test contract

E2e means **real framework execution -> public instrumentation API -> store ->
exported artifact -> Perfetto ingestion/assertions**. Mock callback sequences and
unit-level golden JSON are useful but cannot substitute for these tests.

| Scenario | First required slice | Required verification |
| --- | --- | --- |
| Graph agent + scripted chat model + local tool, sync/async | REF-02 | Correct output, parentage, status, duration, usage, valid artifact |
| Privacy, independent runs, original sizes, bounded/full truncation | REF-01 regressions; REF-02 artifact | No canary leakage in none/metadata; self-contained bounded traces |
| Actual Deep Agent tool loop + file tools + delegated subagent | REF-03 | Proven execution/output, model/tool/subagent spans, usage once |
| Streaming, repeated runs, shared handler | REF-03 | Isolation, bounded token events, stream completion semantics |
| Large/unfinished trace, many sessions, event pressure | REF-04 | Caps, reclamation, loss markers, nonquadratic hot path |
| Dynamic fan-out, nested/repeated nodes, loops, retries | REF-05 | Track separation, containment vs causal links, final root outcome |
| Interrupt/checkpoint/resume, concurrent invocations | REF-05 | Session links, distinct invocations, no fabricated continuous spans |
| Deep Agent cancellation/tool failure + early stream close | REF-06 | Original errors, partial/abandoned status, no worker leaks |
| Store/export/timeouts/close failure injection | REF-06 | No cascade or deadlock, visible loss, bounded lifecycle |
| SQLite reopen/crash/recovery/retention | REF-07 | Equivalent semantic artifact, no orphan records |
| Clean-wheel demos and supported dependency matrix | REF-08 | Works outside checkout without credentials/services |
| CPU/RSS sampling; profile correlation; OTel | Respective optional slice | Measurement/context accuracy and optionality |

Test infrastructure rules:

- Required agent CI installs `deepagents` and supported runtime dependencies;
  missing extras fail that job rather than silently skipping e2e coverage.
- Disable external tracing integrations and block outbound network during tests;
  use no `.env`, real credentials, provider model downloads, or internet tools.
- Fake only external models/tools/clocks/samplers, not the agent runtime. Script
  tool calls and delegation; use temporary/in-memory filesystem/checkpoint backends.
- Assert semantic relationships, outputs, and loss/privacy contracts rather than
  framework-private callback counts, unstable IDs, or exact real-time durations.
- Verify actual concurrency using synchronization primitives, not assumptions that
  an async invocation necessarily overlaps. Use bounded subprocess watchdogs.
- Provision pinned Perfetto tooling in CI ahead of tests; regular offline unit tests
  never download binaries. Artifact ingestion is a required release job, not an
  optional local viewer check. Golden traces contain synthetic data only.
- Provider-backed OpenAI/Ollama demonstrations may remain opt-in manual examples;
  they are not evidence of e2e correctness and are not required CI jobs.

## Tracking checklist and completion policy

- [x] REF-01 — Safety, scope, regression foundation, CI (#14)
- [x] REF-02 — Modern LangGraph-to-Perfetto MVP; legacy removal (M1, #15)
- [x] REF-03 — Real Deep Agents e2e and demo (#16)
- [x] REF-04 — Bounded efficient recording (M2 with REF-03, #17)
- [x] REF-05 — Graph/concurrency/retry/resume semantics (#18)
- [x] REF-06 — Failure isolation and lifecycle (M3 with REF-05, #19)
- [x] REF-07 — SQLite durability and reopen (#20)
- [x] REF-08 — Local-first release (M4; core epic complete, #21)
- [x] REF-09 — Optional CPU/RSS sampling (#22)
- [x] REF-10 — Optional sampling-profiler interoperability / go-no-go (#23; no-go ADR)
- [x] REF-11 — Optional OpenTelemetry export-only bridge (#24)

Every implementation PR names its REF ID, links this parent, lists prerequisites,
and supplies acceptance evidence and runnable verification commands. Run configured
formatter, linter, type checker, tests, and package build. Add required e2e jobs as
their slices land rather than postponing integration testing until REF-08.

Mark slices complete only with code and acceptance evidence, not a plan or a mock.
If child issues are created, mirror their links/status here and in the parent.
Record nonblocking debt with impact, priority, rationale, and completion condition
(existing backlog #12 may be reused). Privacy, corruption, unbounded default memory,
lost application exceptions, and failing required tests cannot be waived as debt.
Changing scope/dependencies requires updating this document and the parent issue.
Core closure may move unfinished optional slices into explicitly linked follow-up
issues; it must not imply those optional capabilities were implemented.
