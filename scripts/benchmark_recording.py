"""Reproducible local benchmark; reports rather than asserting noisy timings."""

import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from langchain_ollie import (
    CapturePolicy,
    MemoryTraceStore,
    RecordingLimits,
    RecordingSession,
    SpanKind,
    TraceRecorder,
)


def measure(callback, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - start)
    return round(statistics.median(samples) * 1000, 3)


def events(count):
    with TraceRecorder(
        MemoryTraceStore(),
        limits=RecordingLimits(
            max_span_events=4096,
            max_trace_events=4096,
        ),
    ) as recorder:
        recorder.start_span("root", "root", SpanKind.GRAPH)
        for _ in range(count):
            recorder.add_event("root", "event")
        recorder.end_span("root")


def traces(count, children):
    with TraceRecorder(MemoryTraceStore()) as recorder:
        for i in range(count):
            recorder.start_span(str(i), "root", SpanKind.GRAPH)
            for j in range(children):
                cid = f"{i}:{j}"
                recorder.start_span(cid, "child", SpanKind.NODE, parent_correlation_id=str(i))
                recorder.end_span(cid)
            recorder.end_span(str(i))


def graph_runs(policy=None, concurrent=False):
    graph = StateGraph(dict)
    graph.add_node("work", lambda state: state)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    app = graph.compile()
    if policy is None:
        for _ in range(50):
            app.invoke({"body": "x" * 200})
        return
    with RecordingSession(capture=policy) as session:

        def invoke(_):
            return app.invoke({"body": "x" * 200}, config={"callbacks": [session.handler]})

        if concurrent:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(invoke, range(50)))
        else:
            list(map(invoke, range(50)))


def main():
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "unit": "median milliseconds / 5 repetitions",
                "events_1000": measure(lambda: events(1000)),
                "events_2000": measure(lambda: events(2000)),
                "small_traces_200": measure(lambda: traces(200, 1)),
                "one_trace_400_children": measure(lambda: traces(1, 400)),
                "graph_50_uninstrumented": measure(graph_runs),
                "graph_50_metadata": measure(lambda: graph_runs(CapturePolicy.metadata())),
                "graph_50_bounded": measure(lambda: graph_runs(CapturePolicy.bounded())),
                "graph_50_full": measure(lambda: graph_runs(CapturePolicy.full())),
                "graph_50_concurrent": measure(lambda: graph_runs(CapturePolicy.metadata(), True)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
