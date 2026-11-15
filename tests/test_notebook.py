"""Execute the shipped notebook's actual cells against the real agent runtime."""

import json
from pathlib import Path
from typing import Any

import pytest

from langchain_ollie import SpanKind

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/01_langgraph_multi_tool_trace.ipynb"


def _execute(monkeypatch, tmp_path, *, viewer: bool = False) -> dict[str, Any]:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "viewer" in cell["metadata"].get("tags", []) and not viewer:
            continue
        source = "".join(cell["source"])
        # Shell/magic cells are manual viewer launchers, not the offline example.
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        exec(compile(source, f"{NOTEBOOK.name}:{cell['id']}", "exec"), namespace)
    return namespace


def test_notebook_runs_real_tools_logs_and_exports_trace(monkeypatch, tmp_path, capsys):
    namespace = _execute(monkeypatch, tmp_path)
    output = capsys.readouterr().out
    assert namespace["final_answer"] == "Total: $45.00 (4 notebooks, 10% discount)."
    names = [span.name for span in namespace["tool_spans"]]
    expected = ("get_unit_price", "calculate_subtotal", "apply_discount")
    assert all(any(name.endswith(tool) for name in names) for tool in expected)
    positions = [output.index(f"TOOL {tool} start") for tool in expected]
    assert positions == sorted(positions)
    assert output.index("TOOL apply_discount result") < output.index("MODEL final answer")
    assert "STREAM completed node: tools" in output
    assert "RUN finished" in output
    artifact = json.loads(namespace["trace_path"].read_text())
    tool_events = [e for e in artifact["traceEvents"] if e.get("cat") == "tool" and e["ph"] == "X"]
    assert len(tool_events) == 3
    assert any(e["ph"] == "C" for e in artifact["traceEvents"])
    assert not artifact["otherData"]["incomplete"]
    assert all(span.parent_span_id for span in namespace["spans"] if span.kind == SpanKind.TOOL)


def test_notebook_schema_and_optional_viztracer_browser_report(monkeypatch, tmp_path, capsys):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("viztracer")
    ipython_display = pytest.importorskip("IPython.display")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    displays = []
    monkeypatch.setattr(ipython_display, "display", displays.append)

    def unexpected_browser_launch(*args, **kwargs):
        raise AssertionError("Run All must not automatically launch a browser")

    monkeypatch.setattr("webbrowser.open_new_tab", unexpected_browser_launch)
    namespace = _execute(monkeypatch, tmp_path, viewer=True)
    assert not any("iframe" in item.data or "srcdoc" in item.data for item in displays)
    output = capsys.readouterr().out
    assert "vizviewer" in output
    assert str(namespace["trace_path"].resolve()) in output
    report = namespace["html_path"].read_text()
    assert all(
        tool in report for tool in ("get_unit_price", "calculate_subtotal", "apply_discount")
    )
    assert "trace-viewer" in report
    assert "file_info" not in namespace["trace_payload"]
    assert len(namespace["trace_payload"]["traceEvents"]) > 0
