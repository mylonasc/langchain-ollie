"""Optional SDK in-memory export; no provider, collector or network needed."""

import argparse

from local_agent import build_agent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from langchain_ollie import RecordingSession
from langchain_ollie.exporters.otel import OpenTelemetryProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="otel-local.json")
    args = parser.parse_args()
    exporter = InMemorySpanExporter()
    with RecordingSession(processors=[OpenTelemetryProcessor(exporter)], strict=True) as session:
        result = build_agent().invoke(
            {"messages": [("user", "add")]}, config={"callbacks": [session.handler]}
        )
    session.export(args.destination)
    print(result["messages"][-1].content)
    print(f"SDK spans: {len(exporter.get_finished_spans())}")


if __name__ == "__main__":
    main()
