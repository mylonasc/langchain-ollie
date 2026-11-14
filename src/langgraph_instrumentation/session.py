"""Small local recording facade with explicit resource ownership."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self, TextIO, runtime_checkable
from uuid import uuid4

from .adapters import LangGraphCallbackHandler
from .exporters import PerfettoExporter
from .limits import RecordingLimits
from .models import TraceBundle, TraceQuery
from .policies import CapturePolicy, RedactionPolicy
from .recorder import SpanProcessor, TraceRecorder
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
        capture_streaming_tokens: bool = False,
        processors: Iterable[SpanProcessor] = (),
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
            processors=processors,
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
            capture_streaming_tokens=capture_streaming_tokens,
        )
        self._finished: tuple[TraceBundle, ...] | None = None
        self._closed = False
        self._finished_diagnostics: dict[str, int] | None = None
        self._finalize_lock = threading.Lock()
        self._finalizing_thread: int | None = None
        self._close_result = True

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
        self.recorder.expire_spans()
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
            return self._close_result
        if self._finalizing_thread == threading.get_ident():
            return False
        start = time.monotonic()
        recorder_ok = self.recorder.close(timeout)
        if not recorder_ok and not self.recorder.is_closed:
            return False
        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
        acquired = (
            self._finalize_lock.acquire()
            if remaining is None
            else self._finalize_lock.acquire(timeout=remaining)
        )
        if not acquired:
            return False
        try:
            if self._closed:
                return self._close_result
            self._finalizing_thread = threading.get_ident()
            if self._finished is None:
                self._finished = self.snapshot()
                self._finished_diagnostics = self.diagnostics
            store_ok = self._store.close(remaining) if self._owns_store else True
            self._closed = store_ok
            self._close_result = recorder_ok and store_ok
            return self._close_result
        finally:
            self._finalizing_thread = None
            self._finalize_lock.release()

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
