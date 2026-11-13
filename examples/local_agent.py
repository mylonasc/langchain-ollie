"""No-network tool-calling LangGraph agent: python examples/local_agent.py trace.json."""

import argparse
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_instrumentation import RecordingSession


@tool
def add(a: int, b: int) -> int:
    """Add two integers locally."""
    return a + b


class ScriptedModel(BaseChatModel):
    """A real chat-model interface with deterministic tool calling, not a provider."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-local-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        self.calls += 1
        if self.calls > 2:
            raise AssertionError("unexpected extra model call")
        if self.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call-add"}],
                usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            )
        else:
            assert isinstance(messages[-1], ToolMessage)
            message = AIMessage(
                content=f"Result: {messages[-1].content}",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_agent() -> Any:
    model = ScriptedModel().bind_tools([add])

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
