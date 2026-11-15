# Resource sampler overhead

Process counters help explain resource activity, but taking samples also consumes
time. Use these measurements when choosing an interval and attempt cap. Enable
sampling only when process-wide evidence is useful; see the
[resource-counter guide](../README.md#optional-process-resource-counters).

## Baseline and interpretation

CPython 3.12.3, Linux x86-64, psutil 7.2.2, same machine as the
[recording benchmark](REF04_BASELINE.md).
200 manual `ProcessSampler.sample_once()` calls against a real source and memory
store: median 48.8 µs, p95 53.9 µs; total sampler time 9.81 ms. No dropped/missing
samples. At the default 100 ms interval this baseline costs roughly 0.05% of one
core, excluding scheduling effects. SQLite commits or a slow custom source can
cost substantially more. No production overhead promise is made.

## Reproduce the measurement

Install the optional resource dependency from a checkout:

```bash
uv sync --group dev --group resources
```

Use manual sampling to measure calls without background scheduling effects:

```python
import statistics
import time

from langchain_ollie import RecordingSession
from langchain_ollie.resources import ProcessSampler

durations = []
with RecordingSession() as recording:
    sampler = ProcessSampler(recording.recorder, max_samples=200)
    sampler.start(background=False)
    try:
        for _ in range(200):
            started = time.perf_counter_ns()
            sampler.sample_once()
            durations.append(time.perf_counter_ns() - started)
    finally:
        sampler.close()

print(f"Median: {statistics.median(durations) / 1_000:.1f} microseconds")
print(f"p95: {sorted(durations)[189] / 1_000:.1f} microseconds")
print(f"Total: {sum(durations) / 1_000_000:.2f} milliseconds")
```

The default source reads only the current process. Keep the store, versions and
machine comparable when repeating the measurement. This measures manual calls;
it does not include background-thread scheduling cost.

## Regression review

Investigate a greater-than-2x median regression on the same machine rather than
gating shared CI on timing. Use recorded overhead and loss to judge whether a
sampling configuration is providing useful evidence:

Each root records interval/cap and final total sampler overhead/missing/dropped
counts, and every successful metric includes read duration in nanoseconds.
For overhead without sampling, see the [recording benchmark](REF04_BASELINE.md).
