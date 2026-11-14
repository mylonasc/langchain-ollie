"""Optional psutil example: process-wide counters alongside an offline agent."""

import argparse

from local_agent import build_agent

from langgraph_instrumentation import RecordingSession
from langgraph_instrumentation.resources import ProcessSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="resources.json")
    args = parser.parse_args()
    with RecordingSession() as session, ProcessSampler(session.recorder):
        build_agent().invoke(
            {"messages": [("user", "add")]}, config={"callbacks": [session.handler]}
        )
    session.export(args.destination)


if __name__ == "__main__":
    main()
