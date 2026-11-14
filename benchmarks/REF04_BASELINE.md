# REF-04 recording benchmark

Run `uv run --no-sync python scripts/benchmark_recording.py` with development
dependencies installed. Five repetitions; median elapsed milliseconds; graph
construction and orderly recorder shutdown are included. No provider calls.

Baseline: CPython 3.12.3, Linux x86-64, glibc 2.39, November 2026. This is a local
comparison, not an across-machine latency promise or CPU attribution.

| Workload | Median ms |
| --- | ---: |
| 1,000 event appends and completion | 16.182 |
| 2,000 event appends and completion | 32.624 |
| 200 small traces, one child each | 23.797 |
| One trace, 400 children | 22.130 |
| 50 uninstrumented minimal graphs | 14.923 |
| 50 graphs, metadata capture | 36.252 |
| 50 graphs, bounded capture | 38.198 |
| 50 graphs, full capture | 37.586 |
| 50 graphs, metadata, four concurrent workers | 40.723 |

Doubling events costs 2.02x, versus the former repeated-span-rebuild path.
Metadata instrumentation adds approximately 0.43 ms per trivial graph invocation
here; this is substantial for microtasks and should not be advertised as free.

Review thresholds on the same machine: event doubling should remain below 3x;
median metadata overhead should remain below 1 ms/invocation for this workload.
Investigate a >25% median regression with repeated runs before accepting a hot-path
change. Timings are not asserted in shared-runner correctness tests. Required tests
instead prove no Span construction during appends, finite byte/record admission,
cooperative expiry, capped token detail, and retained-memory tolerance under 5,000
dropped oversized payloads. Limits bound captured JSON and structural records,
not the application's own memory or an exact Python heap/RSS byte count.
