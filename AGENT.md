# Agent Guide

## Purpose

Guidance for changes to this repository. Architecture boundaries are recorded in
[`docs/architecture.md`](docs/architecture.md); current work is tracked in GitHub
issues. Work against each issue's scope and acceptance criteria.

## Repository Rules

- Use the `src/langgraph_instrumentation/` package for library code.
- Keep examples outside the library package.
- Do not restore prototype import paths; the clean API break is intentional.
- Modern LangGraph/Deep Agents only. Rewrite or drop uncertain legacy framework
  features rather than building compatibility wrappers. Perfetto is primary;
  resource sampling and OTel are optional lower-priority slices. Profiler timeline
  correlation is a documented no-go, not a pending feature.
- Preserve existing unrelated worktree changes. Generated artifacts currently
  present in the worktree must not be silently committed or deleted.
- Never use real credentials or require paid/network LLM calls in tests.
- Treat prompt, response, state, metadata, and tool payloads as potentially
  sensitive.

## Dependency Direction

- Neutral models import no adapter, store implementation, or exporter.
- Stores depend only on neutral models and store-level protocols.
- Exporters depend only on neutral models, exporter protocols, and optional SDKs.
- Framework adapters depend on recorder-facing interfaces, never concrete stores
  or exporters.
- The recorder coordinates stores and processors but contains no Perfetto,
  SQLite, OTLP, or demo-specific behavior.
- Optional dependencies must not be imported by the core package at module
  import time.

## Implementation Practices

- Prefer small protocols and composition over concrete base-class hierarchies.
- Keep public APIs typed and deliberately exported from `__init__.py`.
- Use immutable or snapshot records across component boundaries.
- Sanitize and redact before persistence or export.
- Do not hold recorder/store locks while invoking user code or exporters.
- Make flush and close behavior explicit, bounded, and idempotent.
- Preserve an agent application's original exception if instrumentation also
  fails.
- Use deterministic clocks and ID generators in tests.

## Required Verification

Each feature must add tests at its own abstraction boundary. Before submitting,
run the repository's configured formatter, linter, type checker, unit tests, and
package build. Do not introduce an undocumented replacement tool.

Tests must not depend on:

- OpenAI or Ollama availability
- external OTLP collectors in the default suite
- network access
- local developer environment variables
- callback timing based on real sleeps

## Scoped Agent Guides

Maintain scoped `AGENT.md` files with the directory they govern:

- `src/langgraph_instrumentation/adapters/AGENT.md`
- `src/langgraph_instrumentation/stores/AGENT.md`
- `src/langgraph_instrumentation/exporters/AGENT.md`

The nearest guide supplements this file and documents local protocols, forbidden
dependencies, and the contract or fixture suite implementations must pass. Review
all guides when architecture boundaries change.

## Pull Requests

- Link the issue and state which acceptance criteria are satisfied.
- Identify deferred work explicitly instead of adding speculative compatibility
  or abstraction layers.
- Never use technical debt to bypass correctness, security, data-loss,
  public-contract, or required-test failures.
- Commit completed slices separately and close issues only after their gates pass.
