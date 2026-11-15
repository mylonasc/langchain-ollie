"""Opt-in process-wide counters; psutil is imported only when sampling starts."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self
from uuid import uuid4

from .models import SpanKind
from .recorder import TraceRecorder


@dataclass(frozen=True, slots=True)
class ProcessReading:
    cpu_seconds: float
    rss_bytes: int


class ProcessSource(Protocol):
    """Source for this process only; no child/process discovery or attach."""

    def read(self) -> ProcessReading: ...


class _PsutilSource:
    def __init__(self) -> None:
        try:
            import psutil
        except ImportError:
            raise ImportError("install Ollie with the resources extra to sample") from None
        self._process = psutil.Process(os.getpid())

    def read(self) -> ProcessReading:
        cpu = self._process.cpu_times()
        return ProcessReading(cpu.user + cpu.system, self._process.memory_info().rss)


class ProcessSampler:
    """Bounded, opt-in sampler with a separate process-scope trace root.

    CPU percent uses one logical core as 100%, so multithreaded work can exceed
    100%. Metrics describe the entire process, not the containing agent. Stop the
    sampler before closing its recorder. Sources must return promptly; timeout
    cannot interrupt an uncooperative native/source call.
    """

    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        interval: float = 0.1,
        max_samples: int = 600,
        source: ProcessSource | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(interval, bool) or not math.isfinite(interval) or interval < 0.01:
            raise ValueError("interval must be finite and at least 0.01 seconds")
        if type(max_samples) is not int or not 1 <= max_samples <= 100_000:
            raise ValueError("max_samples must be between 1 and 100000")
        self._recorder, self._source = recorder, source
        self._interval, self._max_samples = interval, max_samples
        self._now = monotonic_ns
        self._correlation = f"process-sampler:{uuid4().hex}"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = self._closed = False
        self._attempts = self._missing = self._dropped = self._overhead_ns = 0
        self._previous: tuple[int, float] | None = None

    @property
    def diagnostics(self) -> dict[str, int]:
        return {
            "attempts": self._attempts,
            "missing": self._missing,
            "dropped": self._dropped,
            "overhead_ns": self._overhead_ns,
        }

    def start(self, *, background: bool = True) -> Self:
        if self._started or self._closed:
            raise RuntimeError("sampler is single-use")
        self._source = self._source or _PsutilSource()
        span = self._recorder.start_span(
            self._correlation,
            "Process resources",
            SpanKind.CUSTOM,
            attributes={
                "measurement.scope": "process",
                "process.pid": os.getpid(),
                "cpu.normalization": "one logical core = 100 percent",
                "sampler.interval_seconds": self._interval,
                "sampler.max_samples": self._max_samples,
            },
        )
        if span is None:
            raise RuntimeError("sampler root admission rejected")
        self._started = True
        if background:
            self._thread = threading.Thread(
                target=self._run, name="local-trace-resources", daemon=True
            )
            self._thread.start()
        return self

    def sample_once(self) -> bool:
        """Manual deterministic sampling; no per-sample buffer outside the recorder."""
        with self._lock:
            if not self._started or self._closed or self._stop.is_set():
                return False
            if self._attempts >= self._max_samples:
                return False
            self._attempts += 1
            start = self._now()
            try:
                assert self._source is not None
                reading = self._source.read()
                if (
                    not math.isfinite(reading.cpu_seconds)
                    or reading.cpu_seconds < 0
                    or type(reading.rss_bytes) is not int
                    or reading.rss_bytes < 0
                ):
                    raise ValueError("invalid process reading")
                now = self._now()
                values: dict[str, int | float] = {
                    "cpu_seconds": reading.cpu_seconds,
                    "rss_bytes": reading.rss_bytes,
                }
                if self._previous is not None:
                    before, cpu = self._previous
                    if now > before and reading.cpu_seconds >= cpu:
                        values["cpu_percent_one_core"] = (
                            (reading.cpu_seconds - cpu) * 1e11 / (now - before)
                        )
                self._previous = now, reading.cpu_seconds
                values["sampler_read_ns"] = max(0, now - start)
                values["missing_samples"] = self._missing
                values["dropped_samples"] = self._dropped
                metric = self._recorder.record_metric(
                    self._correlation,
                    "process.resources",
                    values,
                    attributes={
                        "measurement.scope": "process",
                        "process.pid": os.getpid(),
                        "units": {
                            "cpu_seconds": "s",
                            "rss_bytes": "bytes",
                            "cpu_percent_one_core": "%",
                            "sampler_read_ns": "ns",
                        },
                    },
                )
                if metric is None:
                    self._dropped += 1
                return metric is not None
            except Exception:
                self._missing += 1  # never retain permission/OS error text
                return False
            finally:
                self._overhead_ns += max(0, self._now() - start)

    def _run(self) -> None:
        while not self._stop.is_set() and self._attempts < self._max_samples:
            self.sample_once()
            if self._stop.wait(self._interval):
                return

    def close(self, timeout: float | None = None) -> bool:
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be finite and nonnegative")
        self._stop.set()
        start = time.monotonic()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                return False
        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
        acquired = (
            self._lock.acquire() if remaining is None else self._lock.acquire(timeout=remaining)
        )
        if not acquired:
            return False
        try:
            if not self._closed and self._started:
                self._recorder.end_span(
                    self._correlation,
                    attributes={f"sampler.{key}": value for key, value in self.diagnostics.items()},
                )
            self._closed = True
            return True
        finally:
            self._lock.release()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()
