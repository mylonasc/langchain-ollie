"""Execution adapters that invoke an agent under an active recording session."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from langchain_core.messages import BaseMessage

from ..models import JSONValue
from ..session import RecordingSession
from .models import EvalCase, EvalOutput, EvalTurn

_MAX_DEPTH = 16
_MAX_ITEMS = 2_000
_MAX_NODES = 20_000
_MAX_STRING = 100_000


@dataclass(frozen=True, slots=True)
class EvalInvocation:
    """Everything an adapter needs to execute one turn."""

    run_id: str
    case: EvalCase
    turn: EvalTurn
    turn_index: int
    trial_index: int
    session: RecordingSession

    def __post_init__(self) -> None:
        if not isinstance(self.case, EvalCase):
            raise TypeError("case must be an EvalCase")
        if not isinstance(self.turn, EvalTurn):
            raise TypeError("turn must be an EvalTurn")
        if not isinstance(self.session, RecordingSession):
            raise TypeError("session must be a RecordingSession")
        for name, value in (
            ("turn_index", self.turn_index),
            ("trial_index", self.trial_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Invoke an agent synchronously for one turn."""

    def invoke(self, invocation: EvalInvocation) -> EvalOutput:
        """Execute the turn and return a JSON-safe output."""
        ...


@runtime_checkable
class AsyncExecutionAdapter(Protocol):
    """Invoke an agent asynchronously for one turn."""

    async def ainvoke(self, invocation: EvalInvocation) -> EvalOutput:
        """Execute the turn and return a JSON-safe output."""
        ...


class LangGraphEvalAdapter:
    """Invoke a compiled LangGraph/Deep Agents app with explicit callbacks.

    ``state_mode="accumulate"`` keeps message history in the adapter and sends
    the full history on every turn. ``state_mode="checkpointer"`` sends only the
    turn input and supplies a per-run ``thread_id``; the app must own a checkpointer.
    """

    def __init__(
        self,
        app: Any,
        *,
        config: Mapping[str, object] | None = None,
        state_mode: str = "accumulate",
        text_extractor: Callable[[object], str | None] | None = None,
    ) -> None:
        if state_mode not in ("accumulate", "checkpointer"):
            raise ValueError("state_mode must be 'accumulate' or 'checkpointer'")
        if not hasattr(app, "invoke") and not hasattr(app, "ainvoke"):
            raise TypeError("app must provide invoke() or ainvoke()")
        self._app = app
        self._config = dict(config or {})
        self._state_mode = state_mode
        self._text_extractor = text_extractor
        self._history: dict[tuple[str, str, int], list[object]] = {}

    def invoke(self, invocation: EvalInvocation) -> EvalOutput:
        if not hasattr(self._app, "invoke"):
            raise TypeError("app does not support synchronous invocation")
        payload, config = self._prepare(invocation)
        result = self._app.invoke(payload, config=config)
        return self._output(result)

    async def ainvoke(self, invocation: EvalInvocation) -> EvalOutput:
        if not hasattr(self._app, "ainvoke"):
            raise TypeError("app does not support async invocation")
        payload, config = self._prepare(invocation)
        result = await self._app.ainvoke(payload, config=config)
        return self._output(result)

    def _prepare(self, invocation: EvalInvocation) -> tuple[Mapping[str, object], dict[str, Any]]:
        config: dict[str, Any] = dict(self._config)
        raw_callbacks = config.get("callbacks")
        callbacks = (
            list(cast(Sequence[object], raw_callbacks))
            if isinstance(raw_callbacks, Sequence) and not isinstance(raw_callbacks, (str, bytes))
            else []
        )
        if invocation.session.handler not in callbacks:
            callbacks.append(invocation.session.handler)
        config["callbacks"] = callbacks

        payload: dict[str, Any] = dict(invocation.turn.payload())
        if self._state_mode == "checkpointer":
            raw_configurable = config.get("configurable")
            configurable = (
                dict(cast(Mapping[str, object], raw_configurable))
                if isinstance(raw_configurable, Mapping)
                else {}
            )
            configurable["thread_id"] = (
                f"eval:{invocation.run_id}:{invocation.case.case_id}:{invocation.trial_index}"
            )
            config["configurable"] = configurable
            return payload, config

        key = (invocation.run_id, invocation.case.case_id, invocation.trial_index)
        history = self._history.setdefault(key, [])
        if invocation.turn.messages:
            history.extend(invocation.turn.messages)
            payload = {"messages": list(history)}
        elif "messages" in payload:
            supplied = payload["messages"]
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
                history.extend(supplied)
                payload["messages"] = list(history)
        return payload, config

    def _output(self, result: object) -> EvalOutput:
        text = (
            self._text_extractor(result)
            if self._text_extractor is not None
            else _default_text(result)
        )
        return EvalOutput(value=_json_safe(result), text=text)


def _default_text(result: object) -> str | None:
    if isinstance(result, BaseMessage):
        return _message_text(result)
    if isinstance(result, str):
        return result[:_MAX_STRING]
    if isinstance(result, Mapping):
        messages = result.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) and messages:
            return _message_text(messages[-1])
    return None


def _message_text(message: object) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content[:_MAX_STRING]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        fragments: list[str] = []
        for block in content:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments)[:_MAX_STRING] or None
    return None


def _json_safe(value: object, *, depth: int = 0, budget: list[int] | None = None) -> JSONValue:
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        return "<budget>"
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > _MAX_STRING:
            return value[:_MAX_STRING] + f"<truncated:{len(value) - _MAX_STRING}>"
        if isinstance(value, float) and not math.isfinite(value):
            return "<non-finite>"
        return value
    if depth >= _MAX_DEPTH:
        return "<max-depth>"
    if isinstance(value, BaseMessage):
        return {
            "type": type(value).__name__,
            "content": _json_safe(value.content, depth=depth + 1, budget=budget),
            "name": getattr(value, "name", None),
            "id": getattr(value, "id", None),
            "tool_call_id": getattr(value, "tool_call_id", None),
            "tool_calls": _json_safe(
                getattr(value, "tool_calls", None), depth=depth + 1, budget=budget
            ),
            "usage_metadata": _json_safe(
                getattr(value, "usage_metadata", None), depth=depth + 1, budget=budget
            ),
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                result["<truncated>"] = len(value) - index
                break
            result[str(key)] = _json_safe(item, depth=depth + 1, budget=budget)
        return result
    if isinstance(value, (list, tuple)):
        items = list(value)
        mapped = [_json_safe(item, depth=depth + 1, budget=budget) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            mapped.append(f"<truncated:{len(items) - _MAX_ITEMS}>")
        return mapped
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_safe(item, depth=depth + 1, budget=budget) for item in value),
            key=repr,
        )[:_MAX_ITEMS]
    return {"type": type(value).__name__}
