"""Default tests are hermetic: no provider credentials or external connections."""

import os
import runpy
import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is forbidden in the default test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def local_agent(monkeypatch):
    example = Path(__file__).resolve().parents[1] / "examples" / "local_agent.py"
    monkeypatch.syspath_prepend(str(example.parent))
    return runpy.run_path(str(example))["build_agent"]()


@pytest.fixture
def deep_agent_example(monkeypatch):
    example = Path(__file__).resolve().parents[1] / "examples" / "local_deep_agent.py"
    monkeypatch.syspath_prepend(str(example.parent))
    return runpy.run_path(str(example))


@pytest.fixture
def trace_processor() -> str:
    binary = os.environ.get("PERFETTO_TRACE_PROCESSOR")
    assert binary and Path(binary).is_file(), (
        "Provision pinned tooling with scripts/install_perfetto.py and set PERFETTO_TRACE_PROCESSOR"
    )
    return binary
