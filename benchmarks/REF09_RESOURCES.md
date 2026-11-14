# Resource sampler overhead

CPython 3.12.3, Linux x86-64, psutil 7.2.2, same machine as the recording benchmark.
200 manual `ProcessSampler.sample_once()` calls against a real source and memory
store: median 48.8 µs, p95 53.9 µs; total sampler time 9.81 ms. No dropped/missing
samples. At the default 100 ms interval this baseline costs roughly 0.05% of one
core, excluding scheduling effects. SQLite commits or a slow custom source can
cost substantially more. No production overhead promise is made.

Reproduce: create a RecordingSession and `ProcessSampler(..., max_samples=200)`;
`start(background=False)`, measure 200 calls using `time.perf_counter_ns()`, close
sampler before session. The real source uses `psutil.Process(os.getpid())` only.
Review a >2x median regression on the same machine; do not gate shared CI on timing.
Each root records interval/cap and final total sampler overhead/missing/dropped
counts, and every successful metric includes read duration in nanoseconds.
