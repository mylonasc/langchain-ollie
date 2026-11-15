import json
import subprocess
import sys
import threading

import pytest
from perfetto_support import query

from langchain_ollie import RecordingLimits, RecordingSession
from langchain_ollie.resources import ProcessReading, ProcessSampler


class Source:
    def __init__(self):
        self.cpu = 0.0

    def read(self):
        self.cpu += 0.01
        return ProcessReading(self.cpu, 1024)


def test_deterministic_counters_are_bounded_process_scoped_and_not_agent_cpu():
    ticks = iter(range(0, 1_000_000_000, 10_000_000))
    with RecordingSession(strict=True) as session:
        sampler = ProcessSampler(
            session.recorder, source=Source(), max_samples=2, monotonic_ns=lambda: next(ticks)
        ).start(background=False)
        assert sampler.sample_once()
        assert sampler.sample_once()
        assert not sampler.sample_once()
        assert sampler.close() and sampler.close()
    bundle = session.snapshot()[0]
    assert len(bundle.metrics) == 2
    assert bundle.metrics[1].values["cpu_percent_one_core"] == pytest.approx(100 / 3)
    assert bundle.metrics[0].values["rss_bytes"] == 1024
    assert bundle.metrics[0].attributes["measurement.scope"] == "process"
    assert bundle.spans[0].attributes["sampler.attempts"] == 2


def test_permission_errors_and_recorder_drops_are_counted_without_bodies():
    class Denied:
        def read(self):
            raise PermissionError("PRIVATE-CANARY")

    with RecordingSession(limits=RecordingLimits(max_metrics=1)) as session:
        sampler = ProcessSampler(session.recorder, source=Denied()).start(background=False)
        assert not sampler.sample_once()
        assert sampler.diagnostics["missing"] == 1
        sampler._source = Source()
        assert sampler.sample_once()
        assert not sampler.sample_once()
        assert sampler.diagnostics["dropped"] == 1
        sampler.close()
    assert "PRIVATE-CANARY" not in json.dumps(session.snapshot()[0].to_dict())


def test_default_import_does_not_load_sampler_or_psutil():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import langchain_ollie; "
                "assert 'psutil' not in sys.modules; "
                "assert 'langchain_ollie.resources' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_background_sampler_is_stopped_and_worker_released():
    entered = threading.Event()

    class Signalling(Source):
        def read(self):
            entered.set()
            return super().read()

    with RecordingSession() as session:
        sampler = ProcessSampler(session.recorder, source=Signalling()).start()
        assert entered.wait(5)
        assert sampler.close(timeout=5)
        assert sampler._thread is not None and not sampler._thread.is_alive()
    assert session.snapshot()[0].metrics


def test_real_process_cpu_memory_work_in_subprocess():
    # Isolate allocator/CPU work from the test runner; no scheduler-sensitive sleeps.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from langchain_ollie import RecordingSession
from langchain_ollie.resources import ProcessSampler
with RecordingSession() as session:
    sampler = ProcessSampler(session.recorder).start(background=False)
    assert sampler.sample_once()
    memory = bytearray(8_000_000)
    total = sum(i*i for i in range(500_000))
    assert sampler.sample_once()
    sampler.close()
metrics = session.snapshot()[0].metrics
assert metrics[1].values['cpu_seconds'] >= metrics[0].values['cpu_seconds']
assert metrics[1].values['rss_bytes'] >= 1_000_000
assert metrics[1].values['cpu_percent_one_core'] >= 0
""",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs", [{"interval": 0}, {"max_samples": 0}, {"interval": float("nan")}]
)
def test_invalid_sampler_configuration(kwargs):
    with RecordingSession() as session, pytest.raises(ValueError):
        ProcessSampler(session.recorder, **kwargs)


@pytest.mark.perfetto
def test_process_counters_import_with_units_and_process_identity(trace_processor, tmp_path):
    with RecordingSession() as session:
        sampler = ProcessSampler(session.recorder, source=Source()).start(background=False)
        assert sampler.sample_once()
        sampler.close()
    path = tmp_path / "resources.json"
    session.export(path)
    rows = query(trace_processor, path, "SELECT name FROM counter_track")
    assert any("rss_bytes" in r["name"] for r in rows)
    payload = json.loads(path.read_text())
    assert payload["otherData"]["metric_metadata"][0]["attributes"]["units"]["rss_bytes"] == "bytes"
