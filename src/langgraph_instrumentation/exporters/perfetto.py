"""Deterministic Trace Event JSON for offline Perfetto inspection."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO, TypedDict

from ..models import Span, SpanStatus, TraceBundle, attributes_to_dict


class PerfettoArtifact(TypedDict):
    traceEvents: list[dict[str, Any]]
    otherData: dict[str, Any]


class PerfettoExporter:
    """Export one clock-domain recording. Times are microseconds in Trace Event JSON."""

    def to_dict(
        self, bundles: Sequence[TraceBundle], *, diagnostics: Mapping[str, int] | None = None
    ) -> PerfettoArtifact:
        ordered = sorted(bundles, key=lambda b: b.trace.trace_id.value)
        domains = {b.trace.resource_attributes.get("process.clock.id") for b in ordered}
        if len(domains) > 1:
            raise ValueError("cannot combine recordings from different monotonic clock domains")
        spans = sorted(
            (span for bundle in ordered for span in bundle.spans),
            key=lambda s: (s.start_time_monotonic_ns, s.trace_id.value, s.span_id.value),
        )
        origin = min((s.start_time_monotonic_ns for s in spans), default=0)
        events: list[dict[str, Any]] = []
        tracks: dict[tuple[str, str], list[tuple[int, int]]] = {}
        assigned: dict[tuple[str, str], int] = {}
        next_tid = 1
        for span in spans:
            lane = span.execution_lane
            key = (lane.lane_id if lane else "main", span.kind.value)
            slots = tracks.setdefault(key, [])
            tid = next((tid for tid, end in slots if end <= span.start_time_monotonic_ns), None)
            if tid is None:
                tid = next_tid
                next_tid += 1
                label = f"{lane.name or lane.lane_id if lane else 'main'} / {span.kind.value}"
                events.extend(
                    [
                        self._event("thread_name", "M", tid, args={"name": label}),
                        self._event("thread_sort_index", "M", tid, args={"sort_index": tid}),
                    ]
                )
            end = span.end_time_monotonic_ns
            # An unfinished interval cannot safely share a visual track.
            occupied_until = end if end is not None else 2**127
            slots[:] = [(t, e) for t, e in slots if t != tid]
            slots.append((tid, max(occupied_until, span.start_time_monotonic_ns + 1)))
            assigned[(str(span.trace_id), str(span.span_id))] = tid

        if ordered:
            events.append(
                self._event("process_name", "M", 0, args={"name": ordered[0].trace.service_name})
            )
        by_id = {(str(s.trace_id), str(s.span_id)): s for s in spans}
        for flow_id, span in enumerate(spans, start=1):
            tid = assigned[(str(span.trace_id), str(span.span_id))]
            args = attributes_to_dict(span.attributes)
            args.update(
                {
                    "trace_id": str(span.trace_id),
                    "span_id": str(span.span_id),
                    "parent_span_id": str(span.parent_span_id) if span.parent_span_id else None,
                    "status": span.status.value,
                    "links": [link.to_dict() for link in span.links],
                }
            )
            ts = (span.start_time_monotonic_ns - origin) / 1_000
            if span.is_finished:
                events.append(
                    self._event(
                        span.name,
                        "X",
                        tid,
                        ts=ts,
                        dur=(span.duration_ns or 0) / 1_000,
                        cat=span.kind.value,
                        args=args,
                    )
                )
            else:
                args["recording.incomplete"] = True
                events.append(
                    self._event(span.name, "i", tid, ts=ts, s="t", cat=span.kind.value, args=args)
                )
            for event in span.events:
                events.append(
                    self._event(
                        event.name,
                        "i",
                        tid,
                        ts=(event.time_monotonic_ns - origin) / 1_000,
                        s="t",
                        cat="span.event",
                        args=attributes_to_dict(event.attributes),
                    )
                )
            self._containment(events, span, by_id, assigned, origin, flow_id)
        link_id = len(spans)
        for span in spans:
            for link in span.links:
                source_key = (str(link.trace_id), str(link.span_id))
                source = by_id.get(source_key)
                if source is None or source.start_time_monotonic_ns > span.start_time_monotonic_ns:
                    continue  # external/future endpoints remain metadata, never fabricated arrows
                link_id += 1
                target_key = (str(span.trace_id), str(span.span_id))
                for phase, key, timestamp in (
                    ("s", source_key, source.start_time_monotonic_ns),
                    ("f", target_key, span.start_time_monotonic_ns),
                ):
                    events.append(
                        self._event(
                            f"link:{link.relationship}",
                            phase,
                            assigned[key],
                            ts=(timestamp - origin) / 1000,
                            cat="causal_link",
                            id=link_id,
                            bp="e",
                            args={"relationship": link.relationship},
                        )
                    )
        for bundle in ordered:
            for metric in bundle.metrics:
                tid = assigned.get((str(metric.trace_id), str(metric.span_id)), 0)
                events.append(
                    self._event(
                        metric.name,
                        "C",
                        tid,
                        ts=(metric.time_monotonic_ns - origin) / 1_000,
                        cat="metrics",
                        args=dict(metric.values),
                    )
                )
        loss = dict(sorted((diagnostics or {}).items()))
        return {
            "traceEvents": events,
            "otherData": {
                "recording_schema_version": 1,
                "time_unit": "microseconds",
                "monotonic_origin_ns": origin,
                "clock_domain": next(iter(domains), None),
                "resources": [b.trace.to_dict() for b in ordered],
                "loss": loss,
                "incomplete": any(loss.values())
                or any(not s.is_finished or s.status is SpanStatus.ABANDONED for s in spans),
                "flows": "contains=containment; link:*=explicit evidenced links only",
            },
        }

    @staticmethod
    def _event(name: str, phase: str, tid: int, **fields: object) -> dict[str, object]:
        return {"name": name, "ph": phase, "pid": 1, "tid": tid, **fields}

    def _containment(self, events, span: Span, by_id, assigned, origin: int, flow_id: int) -> None:
        if span.parent_span_id is None:
            return
        parent_key = (str(span.trace_id), str(span.parent_span_id))
        parent = by_id.get(parent_key)
        if parent is None:
            return
        child_key = (str(span.trace_id), str(span.span_id))
        for phase, key, timestamp in (
            ("s", parent_key, parent.start_time_monotonic_ns),
            ("f", child_key, span.start_time_monotonic_ns),
        ):
            events.append(
                self._event(
                    "contains",
                    phase,
                    assigned[key],
                    ts=(timestamp - origin) / 1_000,
                    cat="containment",
                    id=flow_id,
                    bp="e",
                    args={"relationship": "contains"},
                )
            )

    def export(
        self,
        bundles: Sequence[TraceBundle],
        destination: str | os.PathLike[str] | TextIO,
        *,
        diagnostics: Mapping[str, int] | None = None,
        pretty: bool = False,
    ) -> None:
        # Validate/encode before opening the destination. Do not stringify values.
        payload = json.dumps(
            self.to_dict(bundles, diagnostics=diagnostics),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        if not isinstance(destination, (str, os.PathLike)):
            destination.write(payload)
            return
        path = Path(destination)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
