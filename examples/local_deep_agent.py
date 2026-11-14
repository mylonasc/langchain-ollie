"""Real offline Deep Agent with file tools and explicit subagent delegation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from scripted_model import ScriptedModel

from langgraph_instrumentation import RecordingSession

FILE_PATH = "/report.txt"
FINAL_RESPONSE = "Result: 10. Report written and read; worker completed."


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers locally."""
    return a * b


def _reply(*, content: str = "", name: str | None = None, args=None, call_id="") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args or {}, "id": call_id}] if name else [],
        usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )


@dataclass
class OfflineDeepAgent:
    agent: Any
    coordinator: ScriptedModel
    worker: ScriptedModel


def build_deep_agent(*, report: str = "Local report", invocations: int = 1) -> OfflineDeepAgent:
    """Use only virtual state files. No shell or host-filesystem tools are offered."""
    if invocations < 1:
        raise ValueError("invocations must be positive")
    coordinator = ScriptedModel(
        name="coordinator",
        responses=(
            _reply(
                name="write_file", args={"file_path": FILE_PATH, "content": report}, call_id="write"
            ),
            _reply(name="read_file", args={"file_path": FILE_PATH}, call_id="read"),
            _reply(
                name="task",
                args={
                    "subagent_type": "general-purpose",
                    "description": f"Multiply 2 by 5. Context: {report}",
                },
                call_id="delegate",
            ),
            _reply(content=FINAL_RESPONSE),
        )
        * invocations,
        expectations=(
            None,
            ("write_file", FILE_PATH),
            ("read_file", report),
            ("task", "worker-result: 10"),
        )
        * invocations,
    )
    worker = ScriptedModel(
        name="worker",
        responses=(
            _reply(name="multiply", args={"a": 2, "b": 5}, call_id="multiply"),
            _reply(content=f"worker-result: 10. {report}"),
        )
        * invocations,
        expectations=(None, ("multiply", "10")) * invocations,
    )
    agent = create_deep_agent(
        model=coordinator,
        backend=StateBackend(),
        middleware=[FilesystemMiddleware(tools=["write_file", "read_file"])],
        subagents=[
            {
                # Override the automatically added general-purpose worker explicitly.
                "name": "general-purpose",
                "description": "Offline multiplication worker",
                "system_prompt": "Use the supplied local multiplication tool.",
                "model": worker,
                "tools": [multiply],
                "middleware": [FilesystemMiddleware(tools=["read_file"])],
            }
        ],
        name="OfflineDeepAgent",
    )
    return OfflineDeepAgent(agent, coordinator, worker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="deep-agent-trace.json")
    destination = parser.parse_args().destination
    scenario = build_deep_agent()
    with RecordingSession(strict=True) as session:
        result = scenario.agent.invoke(
            {"messages": [("user", "Write a report and delegate multiplication")]},
            config={"callbacks": [session.handler]},
        )
    session.export(destination)
    print(result["messages"][-1].content)
    print(f"Virtual report: {FILE_PATH}")
    print(f"Perfetto artifact: {destination}")


if __name__ == "__main__":
    main()
