"""Default tests are hermetic: no provider credentials or external connections."""

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
def local_agent():
    example = Path(__file__).resolve().parents[1] / "examples" / "local_agent.py"
    return runpy.run_path(str(example))["build_agent"]()
