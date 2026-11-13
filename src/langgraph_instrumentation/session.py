"""Small local recording facade with explicit resource ownership."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, Self, TextIO, runtime_checkable
from uuid import uuid4

from .adapters import LangGraphCallbackHandler
from .exporters import PerfettoExporter
from .limits import RecordingLimits
from .models import TraceBundle, TraceQuery
from .policies import CapturePolicy, RedactionPolicy
from .recorder import TraceRecorder
from .stores import MemoryTraceStore, TraceStore

_CLOCK_DOMAIN = f"{os.getpid()}:{uuid4().hex}"
_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class _StoreDiagnostics(Protocol):
    @property
    def diagnostics(self) -> dict[str, int]: ...


class RecordingSession:
    """Record explicitly attached callbacks; snapshots/export remain usable after close.

    A supplied store must be empty and dedicated to this session. It remains open
    after session shutdown. The default memory store is owned and closed here.
    """

    def __init__(
        self,
        *,
        store: TraceStore | None = None,
        capture: CapturePolicy | None = None,
        redaction: RedactionPolicy | None = None,
        limits: RecordingLimits | None = None,
        service_name: str = "local-agent",
        strict: bool = False,
    ) -> None:
        self._limits = limits or RecordingLimits()
        self._strict = strict
        self._store = store if store is not None else MemoryTraceStore()
        self._owns_store = store is None
        if self._store.list_traces(TraceQuery(limit=1)):
            raise ValueError("a recording session requires an empty dedicated store")
        self.recorder = TraceRecorder(
            self._store,
            limits=self._limits,
            owns_store=False,
            strict=strict,
            service_name=service_name,
            resource_attributes={
                "process.pid": os.getpid(),
                "process.clock.id": f"{os.getpid()}:{_CLOCK_DOMAIN}",
            },
        )
        self.handler = LangGraphCallbackHandler(
            self.recorder,
            capture=capture,
            redaction=redaction,
            raise_error=strict,
        )
        self._finished: tuple[TraceBundle, ...] | None = None
        self._closed = False
        self._finished_diagnostics: dict[str, int] | None = None

    @property
    def diagnostics(self) -> dict[str, int]:
        if self._finished_diagnostics is not None:
            return dict(self._finished_diagnostics)
        result = self.recorder.diagnostics
        if isinstance(self._store, _StoreDiagnostics):
            result.update({f"store.{key}": value for key, value in self._store.diagnostics.items()})
        return result

    def snapshot(self) -> tuple[TraceBundle, ...]:
        if self._finished is not None:
            return self._finished
        result = []
        for summary in self._store.list_traces(TraceQuery(limit=self._limits.max_spans)):
            bundle = self._store.get_trace(summary.trace_id)
            if bundle is not None:
                result.append(bundle)
        return tuple(result)

    def export(self, destination: str | Path | TextIO, *, pretty: bool = False) -> None:
        PerfettoExporter().export(
            self.snapshot(),
            destination,
            diagnostics=self.diagnostics,
            pretty=pretty,
        )

    def close(self, timeout: float | None = None) -> bool:
        if self._closed:
            return True
        if self._finished is None:
            if not self.recorder.close(timeout):
                return False
            self._finished = self.snapshot()
            self._finished_diagnostics = self.diagnostics
        if self._owns_store:
            self._closed = self._store.close(timeout)
            return self._closed
        self._closed = True
        return True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        try:
            closed = self.close()
        except Exception:
            if exc_type is None and self._strict:
                raise
            _LOGGER.warning("recording session shutdown failed")
        else:
            if not closed:
                if exc_type is None and self._strict:
                    raise RuntimeError("recording did not close successfully")
                _LOGGER.warning("recording session shutdown incomplete; see diagnostics")
        return False
