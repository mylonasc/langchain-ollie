# Exporter guide

Exporters depend only on neutral records and optional SDKs. Never query stores
or import framework adapters. Validate values before writing; never use
`default=str`. Path writes use atomic replacement and clean up their own temporary
files. File-object writes cannot promise atomicity.

REF-02 requires deterministic output, overlap-safe tracks, clearly labeled
containment (not graph causality), resource/clock/loss metadata, privacy regressions
and real-agent artifacts imported by the pinned Perfetto Trace Processor. The
default unit suite must not download tools. Required artifact CI provisions the
binary explicitly before tests. REF-05 adds causal links only when evidenced.
