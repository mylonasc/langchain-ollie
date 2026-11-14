"""Deterministic external-model substitute shared by the offline agent examples.

This is example/test support, not a production model integration. The actual
LangGraph/Deep Agents runtimes and their callback dispatch remain unmodified.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from threading import Lock
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


class ScriptedModel(BaseChatModel):
    """Consume a finite script; validate tool binding and prior tool results.

    Usage is emitted once, on the last streaming chunk. Each returned message
    is copied so runtime mutations cannot change later invocations of the script.
    """

    responses: tuple[AIMessage, ...]
    expectations: tuple[tuple[str, str] | None, ...] = ()
    _calls: int = PrivateAttr(default=0)
    _bound_tools: frozenset[str] = PrivateAttr(default=frozenset())
    _lock: Any = PrivateAttr(default_factory=Lock)

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def bound_tools(self) -> frozenset[str]:
        with self._lock:
            return self._bound_tools

    @property
    def _llm_type(self) -> str:
        return "scripted-local-model"

    def _get_invocation_params(self, stop=None, **kwargs) -> dict[str, Any]:
        # Never put scripted application bodies in framework invocation metadata.
        return {"model": self.name or "ScriptedModel", "stop": stop}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = frozenset(convert_to_openai_tool(tool)["function"]["name"] for tool in tools)
        with self._lock:
            self._bound_tools = names
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        with self._lock:
            index = self._calls
            if index >= len(self.responses):
                raise AssertionError("unexpected extra scripted model call")
            if self.expectations:
                expected = self.expectations[index]
                if expected is not None:
                    name, content = expected
                    previous = messages[-1]
                    if not isinstance(previous, ToolMessage) or previous.name != name:
                        raise AssertionError("script did not receive the expected tool result")
                    if not isinstance(previous.content, str) or content not in previous.content:
                        raise AssertionError("tool result did not match the scripted expectation")
            response = self.responses[index].model_copy(deep=True)
            if any(call["name"] not in self._bound_tools for call in response.tool_calls):
                raise AssertionError("script requested a tool the runtime did not bind")
            self._calls += 1
            return response

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        response = self._next(messages)
        if response.content:
            if not isinstance(response.content, str):
                raise AssertionError("the example script supports text content only")
            midpoint = max(1, len(response.content) // 2)
            for text in (response.content[:midpoint], response.content[midpoint:]):
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        for index, call in enumerate(response.tool_calls):
            arguments = json.dumps(call["args"], separators=(",", ":"))
            midpoint = max(1, len(arguments) // 2)
            for position, fragment in enumerate((arguments[:midpoint], arguments[midpoint:])):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"] if position == 0 else None,
                                "id": call["id"] if position == 0 else None,
                                "args": fragment,
                                "index": index,
                            }
                        ],
                    )
                )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=response.usage_metadata,
                chunk_position="last",
            )
        )

    async def _astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk
