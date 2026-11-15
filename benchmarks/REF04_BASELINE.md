# Recording benchmark

Instrumentation adds work to every recorded operation. Use this benchmark to
assess whether that cost is acceptable for your workload and to detect regressions
when changing recording code. For capture options, see
[recording limits](../README.md#safety-and-recording-limits).

## Reproduce the measurement

Run from the repository root after installing development dependencies:

```bash
uv sync --group dev
uv run --no-sync python scripts/benchmark_recording.py
```

The benchmark uses five repetitions and reports median elapsed milliseconds.
Graph construction and orderly recorder shutdown are included. It makes no
provider calls. Compare repeated runs on the same machine, not isolated timings
from different hosts.

## Baseline and interpretation

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

Doubling event appends costs 2.02x in this baseline, close to linear growth.
Metadata instrumentation adds approximately 0.43 ms per trivial graph invocation
here; this is substantial for microtasks and should not be advertised as free.

## Regression review

Use these thresholds to decide when to investigate a change, not as portable
latency guarantees:

Review thresholds on the same machine: event doubling should remain below 3x;
median metadata overhead should remain below 1 ms/invocation for this workload.
Investigate a >25% median regression with repeated runs before accepting a hot-path
change. Timings are not asserted in shared-runner correctness tests. Required tests
instead prove no Span construction during appends, finite byte/record admission,
cooperative expiry, capped token detail, and retained-memory tolerance under 5,000
dropped oversized payloads. Limits bound captured JSON and structural records,
not the application's own memory or an exact Python heap/RSS byte count.

If you also enable process counters, measure their additional cost using the
[resource sampler benchmark](REF09_RESOURCES.md). Before contributing a change,
run the [correctness and package checks](../README.md#development-and-artifact-verification)
as well as the benchmark.
