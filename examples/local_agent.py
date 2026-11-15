"""No-network tool-calling LangGraph agent: python examples/local_agent.py trace.json."""

import argparse
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from scripted_model import ScriptedModel

from langchain_ollie import RecordingSession


@tool
def add(a: int, b: int) -> int:
    """Add two integers locally."""
    return a + b


def build_agent() -> Any:
    model = ScriptedModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call-add"}],
                usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            ),
            AIMessage(
                content="Result: 5",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            ),
        ),
        expectations=(None, ("add", "5")),
    ).bind_tools([add])

    def call_model(state: MessagesState, config: RunnableConfig):
        return {"messages": [model.invoke(state["messages"], config=config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile(name="LocalAgent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="trace.json")
    destination = parser.parse_args().destination
    with RecordingSession() as session:
        result = build_agent().invoke(
            {"messages": [("user", "Add 2 and 3")]},
            config={"callbacks": [session.handler]},
        )
    session.export(destination)
    print(result["messages"][-1].content)
    print(f"Perfetto artifact: {destination}")


if __name__ == "__main__":
    main()
