"""Offline SQL assertions shared by actual-agent artifact tests."""

import csv
import io
import subprocess
from collections.abc import Sequence
from pathlib import Path

from langgraph_instrumentation import TraceBundle


def query(binary: str, artifact: Path, sql: str) -> list[dict[str, str]]:
    result = subprocess.run(
        [binary, str(artifact), "-Q", sql],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def assert_imported_spans(binary: str, artifact: Path, bundles: Sequence[TraceBundle]) -> None:
    imported = query(
        binary,
        artifact,
        (
            "SELECT dur, extract_arg(arg_set_id,'args.trace_id') AS trace_id, "
            "extract_arg(arg_set_id,'args.span_id') AS span_id, "
            "extract_arg(arg_set_id,'args.parent_span_id') AS parent_span_id FROM slice "
            "WHERE dur >= 0 AND extract_arg(arg_set_id,'args.span_id') IS NOT NULL"
        ),
    )
    by_id = {(row["trace_id"], row["span_id"]): row for row in imported}
    spans = [span for bundle in bundles for span in bundle.spans]
    assert len(by_id) == len(spans)
    for span in spans:
        row = by_id[(str(span.trace_id), str(span.span_id))]
        assert span.duration_ns is not None
        assert abs(int(row["dur"]) - span.duration_ns) <= 1
        assert row["parent_span_id"] == (
            str(span.parent_span_id) if span.parent_span_id else "[NULL]"
        )
    flows = query(binary, artifact, "SELECT count(*) AS count FROM flow")
    contexts = {(s.trace_id, s.span_id): s for s in spans}
    explicit = sum(
        (link.trace_id, link.span_id) in contexts
        and contexts[(link.trace_id, link.span_id)].start_time_monotonic_ns
        <= s.start_time_monotonic_ns
        for s in spans
        for link in s.links
    )
    assert (
        int(flows[0]["count"]) == sum(span.parent_span_id is not None for span in spans) + explicit
    )
    errors = query(
        binary, artifact, "SELECT name, value FROM stats WHERE severity = 'error' AND value > 0"
    )
    assert not errors
