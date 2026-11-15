# Interactive agent tracing

Use [the multi-tool walkthrough](01_langgraph_multi_tool_trace.ipynb) to follow a
real LangGraph tool-calling loop from live application logs to an interactive
trace, without model-provider credentials.

## Run the notebook

Install the optional notebook dependencies from the repository root, then start
JupyterLab:

```bash
uv sync --group dev --group notebooks
uv run --no-sync jupyter lab notebooks/01_langgraph_multi_tool_trace.ipynb
```

Select the project environment's Python kernel and run the cells in order. The
notebook defines its own scripted chat model and three local tools, so it does
not depend on imports from the repository's example scripts. Agent execution
makes no network or paid model calls.

## What to look for

The agent looks up a unit price, calculates a subtotal and applies a discount.
The execution cell shows model decisions, tool start/results and streamed graph
updates as they happen. It finishes with a total of **$45.00**.

Ollie records the graph, model and tool operations. The last cells export the
trace, check the recorded tool spans, and create a VizTracer HTML report to open
in your browser. VizTracer is used as a viewer for Ollie's Trace Event JSON, not
as a second profiler; no profile/timeline alignment is attempted.

## Open the report in a browser

Use the last cell's report link, or copy its printed `file://` URI into a browser.
If VS Code opens the file in an editor, open the HTML path from your file manager
instead. Browser viewing is the default: the notebook no longer embeds the
timeline, and it does not automatically launch browsers or background servers.

The additional shell cells are manual launchers: run them only when you want to
start a viewer or download the Perfetto helper. Automated checks skip shell/magic
cells and execute the Python walkthrough in a disposable copy, preserving the
notebook's saved outputs. VizTracer 1.1.1's `vizviewer` binds to all network
interfaces; for private traces, prefer the loopback-only command below.

For an HTTP link, run the terminal command printed by the last cell. It serves
only the generated run directory and binds explicitly to `127.0.0.1`:

```bash
python -m http.server 9001 --bind 127.0.0.1 --directory "/path/to/the/run-directory"
```

Open <http://127.0.0.1:9001/agent_trace.html> and stop the server with **Ctrl+C**
when finished. If port 9001 is already occupied, choose another port in both the
command and URL. For remote SSH, containers or Codespaces, use private VS Code
port forwarding; the generated file lives on the kernel's machine. Do not make
the server publicly accessible.

Click in the viewer, use **W/S** to zoom and **A/D** to pan, then select a tool
slice to inspect its attributes. Durations measure elapsed time, not CPU time.
The scripted token counts illustrate usage counters, not real provider costs.

## Keep artifacts and data under your control

Each run writes JSON and HTML into a new subdirectory of `notebook-artifacts/`
relative to the kernel's working directory. These generated files are ignored by
Git. The notebook deliberately enables bounded capture for synthetic catalog and
arithmetic data; use metadata-only capture for sensitive applications unless you
have reviewed what content will be retained.

The report is active JavaScript. Only trust notebooks and reports you have
reviewed. VizTracer's HTML viewer loads compatibility and highlighting scripts
from public CDNs; opening it in a separate browser does not make it fully offline.
If those CDNs are blocked, load the same JSON into an approved
[offline Perfetto viewer](https://perfetto.dev/docs/visualization/offline).
Do not use external viewer scripts with private trace content without approval.

Continue with [capture limits](../README.md#safety-and-recording-limits),
[the recording architecture](../docs/architecture.md), or
[evaluation checks](../docs/evals.md) to turn an observed execution pattern into
a regression test.
