# Adapter Guide

## Scope

This directory translates framework callbacks into recorder operations. Adapters
may import LangChain/LangGraph APIs, neutral models, capture policies, and the
recorder interface. They must not import concrete stores, processors, exporters,
or presentation formats.

## LangGraph Contract

- Support the ordinary `BaseCallbackHandler.on_*` signatures from
  `langchain-core>=1.8,<1.9`; those hooks are used for both synchronous and
  asynchronous LangGraph runs. Do not add `aon_*` methods unless a future
  supported LangChain callback API defines and dispatches them.
- Normalize callback UUIDs to canonical strings before recorder calls. Preserve
  parent IDs, LangGraph metadata, model identifiers, and tool-call identifiers
  as safe attributes.
- Classify LangGraph root chains as graphs and chains carrying
  `langgraph_node` metadata as nodes. Other chain callbacks remain chains.
- Project every application payload with `CapturePolicy`, `MessageProjector`,
  or `MetadataProjector` before it crosses the recorder boundary. Extract token
  usage with `UsageExtractor` and let the recorder own cumulative totals.
- Detailed streaming token events are disabled by default; chunk counts/offsets
  are metadata-only summaries. Detail is opt-in and bounded by recorder budgets.
- Keep no active-run map, lifecycle lock, token total, lane registry, store, or
  exporter in an adapter. The recorder owns correlation and lifecycle state.
- Projection is a single bounded traversal, with framework normalization in
  `projection.py`. No cross-recording message deduplication or payload cache.
- Exceptions retain only type by default. Message capture requires separate
  opt-in plus a content policy; internal logs must never format exceptions.
- Classic LLM/retriever hooks not required by modern agent e2e are dropped.
  Both `ls_integration="langgraph"` and `"deepagents"` identify modern root graphs.
  REF-03 tests actual Deep Agents tools/delegation/streaming, not callback mocks.
- Error callbacks must preserve the application's original exception if
  instrumentation also fails. Adapter callback failures are non-fatal by
  default; opt-in `raise_error` is for diagnostics outside application-error
  handling.

## Tests

Test translation directly with a fake recorder, then run deterministic,
no-network sync and async LangGraph graphs. Cover root/node parentage,
model/tool callbacks, fan-out/fan-in, cancellation, projection, and
streaming configuration.
