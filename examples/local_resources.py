"""Optional psutil example: process-wide counters alongside an offline agent."""

import argparse

from local_agent import build_agent

from langgraph_instrumentation import RecordingSession
from langgraph_instrumentation.resources import ProcessSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="resources.json")
    args = parser.parse_args()
    with RecordingSession() as session:
        sampler = ProcessSampler(session.recorder, interval=0.01).start()
        try:
            sampler.sample_once()  # synchronous baseline for a CPU-percent delta
            build_agent().invoke(
                {"messages": [("user", "add")]}, config={"callbacks": [session.handler]}
            )
            sampler.sample_once()  # guarantee a final observation even for short agents
        finally:
            sampler.close()
    session.export(args.destination)


if __name__ == "__main__":
    main()
