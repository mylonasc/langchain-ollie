# ADR-001: no-go for automatic py-spy/Speedscope timeline correlation

Status: accepted no-go; 2026-11-14. No sampling engine or production
attach integration is added. Revisit only with evidenced timestamp/process domains.

## Why this decision matters

Execution traces show how long operations take, while sampled profiles help
explain which code was running. Combining them on one timeline would be useful
only if samples could be assigned to spans reliably. The reviewed profile format
does not preserve enough timing information to make that assignment.

Use [local traces](../../README.md#view-locally) for operation structure and
[process counters](../../README.md#optional-process-resource-counters) for aligned
process-wide resource measurements. A separately collected stack profile remains
a companion view, not evidence of exact per-task CPU usage.

## Spike evidence

The important question is whether exported samples preserve their positions in
elapsed time. The format review and ambiguity test below show why a shared start
time alone cannot recover those positions.

Reviewed maintained [py-spy documentation](https://github.com/benfred/py-spy), its
[`src/speedscope.rs`](https://github.com/benfred/py-spy/blob/master/src/speedscope.rs),
and the [Speedscope schema](https://www.speedscope.app/file-format-schema.json).
The reviewed serializer's `SpeedscopeFile::new` assigns every thread
`startValue=0`, `endValue=samples.len()/sample_rate`, and each weight
`1/sample_rate`. It does not emit each sample's wall/monotonic timestamp, recording
clock UUID, or a robust process-start identity. Idle filtering, missed samples,
thread start offsets and sampling error can remove wall-clock gaps. A start-time
anchor alone cannot reconstruct those gaps. The format also permits non-time units.

[The ambiguity test](../../tests/test_profiler_spike.py) provides two distinct elapsed-time schedules which
serialize to identical sampled profiles. Even a common start anchor cannot tell
which span contained the second sample. This evidence comes from format analysis
and a synthetic test, not a live profiler-fidelity or overhead measurement.

## Decision

**No-go** for automatically importing these stacks onto Perfetto execution tracks,
exact per-task CPU attribution, or promising cross-process monotonic alignment.
An SDK conversion cannot recover missing information. Adding a bespoke sampler
would duplicate maintained tools and expand privacy, ABI and lifecycle risks.
Keep the existing Perfetto artifact and a clearly separate Speedscope companion.
Process counters remain the supported time-aligned resource path. This avoids
presenting an ambiguous overlay as measured execution evidence.

Reopen this ADR only for a maintained profiler format with real sample timestamps,
an explicit process identity and measured clock-domain anchors/uncertainty. Future
support must cap input bytes, samples, stack depth/frames and sanitized path/name
lengths, reject unsupported units/platforms, test drift and privacy, and measure
profiling plus conversion overhead. A correlation fixture is not sufficient proof
of a live profiler's fidelity.

## Reproducible external workflow (user-run only)

If you need stack-level insight, collect a separate profile only for a workload
you are authorized to inspect. This workflow does not add profile collection or
parsing to the library.

For a disposable, explicitly authorized local workload, install py-spy separately
in an isolated tool environment and consult `py-spy record --help`. A representative
command, **not invoked by this library or its tests**, is:

```bash
py-spy record --format speedscope --rate 50 --duration 10 -o profile.json -- \
  python examples/local_resources.py resources.json
```

Use a longer local synthetic workload if the demo ends before enough samples are
collected. Open `resources.json` in offline Perfetto and `profile.json` in an offline
Speedscope viewer. These are companion views, **not synchronized timelines**. The
10-second/50-Hz settings bound intended samples per thread (~500), not total file
size, stack depth or thread count. Apply an OS file-size/resource limit appropriate
to the platform; discard oversized output before loading/sharing. No parser for
untrusted profiler files ships here. Stack names and filenames can contain secrets;
keep output private or sanitize using reviewed tooling. Never enable locals.

Measure repeated baseline vs profiled runtime and peak RSS on the same machine;
report versions, workload, rate, duration, errors and sampling mode. No py-spy
overhead number is claimed here because no live profiling measurement was made.
The core does not load a profiler or correlator. Apply the
[artifact-handling guidance](../../SECURITY.md#protecting-recordings) to both outputs.

## Platform limits

Profiler permissions and runtime support are separate from tracing support.
Do not weaken host security settings to make an optional companion profile work.

Upstream advertises recent CPython on Linux/macOS/Windows/FreeBSD, but ABI/build,
architecture and permission details matter. Linux containers often deny memory
inspection; macOS generally requires privilege and system Python may be protected.
This library neither escalates nor changes those policies. Unsupported/denied
environments should use ordinary local traces/counters and stop the profiling
attempt, not silently enable a fallback sampler. No free-threaded/PyPy support or
direct Perfetto stack import is claimed. Upstream support is not our tested matrix.
