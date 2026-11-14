# Post-merge issue review — 2026-11-14

The REF implementation was fast-forward merged and pushed to `main` at `67d05ee`.
All 11 post-merge CI jobs passed in [run 43583712207](https://github.com/mylonasc/langgraph_instrumentation/actions/runs/43583712207).
This review changes planning and issue descriptions, not library behavior or the
scope of the completed implementation. No package publication was performed.

## Reconciled historical issues

| Issue | Decision | Evidence / rationale |
| --- | --- | --- |
| #1 | Closed as superseded | REF epic #13 replaced the old architecture roadmap; its original details are not all claimed as delivered. |
| #7 | Rewritten and closed as completed | Modern callback adapter, real sync/async agents, delegation/streaming tests; REF-01/03/05. Separate async hook shims are unnecessary. |
| #8 | Rewritten and closed as completed | Deterministic atomic Perfetto export, real importer tests; REF-02/05. Containment and explicit causality remain separate. |
| #9 | Rewritten and closed as completed | Transactional SQLite v1, shared contracts, reopen/recovery/CLI; REF-07. Unknown schema rejection is not automatic migration support. |
| #10 | Rewritten and closed as superseded | REF-11 uses SDK export-only conversion/batching, not a custom transport stack or full distributed-context integration. Convention/metrics gaps have separate evaluation tickets. |
| #11 | Rewritten and closed as completed | Offline examples, typed package, security/upgrade docs, clean wheels and required compatibility/importer/OTel CI; REF-08/11. Remote model demos and publication are not required. |
| #12 | Rewritten and retained as index | Old debt entries extracted into focused issues; resolved and intentionally excluded items explicitly recorded. |

Already-closed #2–#6 and #13–#24 remain closed. Issue history and git history retain
the earlier plans; the new descriptions explain their disposition instead of
leaving stale checklists actionable.

## Retained backlog

| Issue | Priority / readiness | Expected outcome |
| --- | --- | --- |
| #25 | P2 / ready | Upgrade CI to reviewed Node-24-native actions; preserve all 11 jobs and remove deprecation annotations. |
| #26 | P3 / measure first | Benchmark slow destinations and decide whether bounded buffering is justified; design explicit ordering, loss and lifecycle semantics before implementation. |
| #28 | P3 / design only | Evaluate a versioned opt-in GenAI mapping without changing privacy defaults or implying unsupported compliance. |
| #27 | P4 / demand-gated | Consider optional PostgreSQL only for a demonstrated shared-storage use case, with store-contract and operational requirements. |
| #29 | P4 / demand-gated design | Define SDK metrics aggregation, units, temporality, cardinality and process scope before building a separate optional metrics path. |

P2 is concrete maintenance; P3 is nonblocking evaluation; P4 is a deferred idea,
not committed release scope. Evaluation may conclude with an evidenced no-go.

## Not retained as implementation debt

- Prototype compatibility and separate `aon_*` hooks: contrary to the supported modern callback API and intentional clean break.
- Homemade OTel batching/retry/transport: duplicates the SDK facilities already in use.
- Provider-dependent demos as acceptance gates: unnecessary for the offline-first workflow.
- Automatic profiler timeline correlation: ADR-001 records an evidenced no-go; missing sample timestamps cannot be reconstructed.
- Full distributed propagation: explicitly outside ADR-002. A concrete consumer requirement should precede a separate design proposal.
- Temporary prototype static-analysis exclusions (TD-004): resolved through prototype removal and release gates.

All retained issues now state motivation, scope, non-goals, readiness and observable
acceptance criteria. The review did not start implementation of these follow-ups.
