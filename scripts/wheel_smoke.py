"""Run with a clean wheel interpreter OUTSIDE the checkout; deny all network."""

import importlib.resources
import json
import os
import runpy
import socket
import sys
from pathlib import Path

import langchain_ollie as instrumentation


def blocked(*args, **kwargs):
    raise AssertionError("wheel quickstart attempted network access")


def main():
    root = Path(sys.argv[1]).resolve()
    assert not Path(instrumentation.__file__).is_relative_to(root)
    assert importlib.resources.files(instrumentation).joinpath("py.typed").is_file()
    for name in list(os.environ):
        if name.endswith("API_KEY"):
            del os.environ[name]
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    sys.path.insert(0, str(root / "examples"))
    for example, args in (
        ("local_agent.py", ["graph.json"]),
        ("local_deep_agent.py", ["deepagent.json"]),
        ("local_sqlite.py", ["agent.db", "sqlite.json"]),
    ):
        sys.argv = [example, *args]
        runpy.run_path(str(root / "examples" / example), run_name="__main__")
    for path in ("graph.json", "deepagent.json", "sqlite.json"):
        payload = json.loads(Path(path).read_text())
        assert payload["traceEvents"] and not payload["otherData"]["incomplete"]
    store = instrumentation.SQLiteTraceStore("agent.db", read_only=True)
    trace_id = str(store.list_traces()[0].trace_id)
    store.close()
    for args in (
        ("list", "agent.db"),
        ("summary", "agent.db", trace_id),
        ("export", "agent.db", trace_id, "cli.json"),
    ):
        sys.argv = ["langchain_ollie", *args]
        runpy.run_module("langchain_ollie", run_name="__main__")


if __name__ == "__main__":
    main()
