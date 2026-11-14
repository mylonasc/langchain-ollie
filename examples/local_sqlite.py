"""Offline graph recording, database close/reopen and Perfetto export."""

import argparse

from local_agent import build_agent

from langgraph_instrumentation import PerfettoExporter, RecordingSession, SQLiteTraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", help="new dedicated database path")
    parser.add_argument("destination", help="Perfetto JSON destination")
    args = parser.parse_args()
    store = SQLiteTraceStore(args.database)
    try:
        with RecordingSession(store=store, strict=True) as recording:
            result = build_agent().invoke(
                {"messages": [("user", "Add 2 and 3")]},
                config={"callbacks": [recording.handler]},
            )
        trace_id = recording.snapshot()[0].trace.trace_id
    finally:
        store.close()
    reopened = SQLiteTraceStore(args.database, read_only=True)
    try:
        bundle = reopened.get_trace(trace_id)
        assert bundle is not None
        PerfettoExporter().export([bundle], args.destination, diagnostics=reopened.diagnostics)
    finally:
        reopened.close()
    print(result["messages"][-1].content)
    print(f"Trace ID: {trace_id}")


if __name__ == "__main__":
    main()
