"""LangChain/LangGraph callback translation into recorder operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError as FuturesCancelledError
from functools import wraps
from typing import Any, Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from ..models import Attributes, SpanKind, SpanStatus
from ..policies import (
    CapturePolicy,
    MetadataProjector,
    RedactionPolicy,
    UsageExtractor,
)
from .projection import normalize_framework_value

_LOGGER = logging.getLogger(__name__)
_LANGGRAPH_METADATA_KEYS = frozenset(
    {
        "agent_name",
        "checkpoint_id",
        "graph_name",
        "langgraph_checkpoint_id",
        "langgraph_checkpoint_ns",
        "langgraph_node",
        "langgraph_path",
        "langgraph_step",
        "langgraph_triggers",
        "thread_id",
    }
)


def _safe_callback(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Contain projection failures before the framework can log their bodies."""

    @wraps(callback)
    def guarded(self: LangGraphCallbackHandler, *args: Any, **kwargs: Any) -> Any:
        try:
            return callback(self, *args, **kwargs)
        except Exception:
            _LOGGER.warning("LangGraph callback translation failed")
            if self.raise_error:
                raise
            return None

    return guarded


class _Recorder(Protocol):
    def start_span(
        self,
        correlation_id: str,
        name: str,
        kind: SpanKind,
        *,
        parent_correlation_id: str | None = None,
        attributes: Attributes | None = None,
        lane_id: str | None = None,
        lane_name: str | None = None,
        trace_name: str | None = None,
    ) -> object | None: ...

    def add_event(
        self,
        correlation_id: str,
        name: str,
        *,
        attributes: Attributes | None = None,
    ) -> object | None: ...

    def record_metric(
        self,
        correlation_id: str,
        name: str,
        values: Mapping[str, int | float],
        *,
        attributes: Attributes | None = None,
        cumulative: bool = False,
    ) -> object | None: ...

    def end_span(
        self,
        correlation_id: str,
        *,
        status: SpanStatus = SpanStatus.SUCCESS,
        status_description: str | None = None,
        attributes: Attributes | None = None,
    ) -> object | None: ...


class LangGraphCallbackHandler(BaseCallbackHandler):
    """Translate LangChain callbacks into a stateless recorder-facing API.

    LangChain dispatches these ordinary synchronous hooks for both synchronous
    and asynchronous executions. Payload capture is metadata-only by default,
    and per-token events are opt-in.
    """

    def __init__(
        self,
        recorder: _Recorder,
        *,
        capture: CapturePolicy | None = None,
        redaction: RedactionPolicy | None = None,
        metadata_projector: MetadataProjector | None = None,
        usage_extractor: UsageExtractor | None = None,
        capture_streaming_tokens: bool = False,
        capture_exception_messages: bool = False,
        raise_error: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(capture_streaming_tokens, bool):
            raise TypeError("capture_streaming_tokens must be a boolean")
        if not isinstance(raise_error, bool):
            raise TypeError("raise_error must be a boolean")
        if not isinstance(capture_exception_messages, bool):
            raise TypeError("capture_exception_messages must be a boolean")
        self._recorder = recorder
        self._capture = capture or CapturePolicy()
        self._redaction = redaction or RedactionPolicy()
        self._metadata = metadata_projector or MetadataProjector(
            include_keys=_LANGGRAPH_METADATA_KEYS,
            redaction=self._redaction,
        )
        self._usage = usage_extractor or UsageExtractor()
        self._capture_streaming_tokens = capture_streaming_tokens
        self._capture_exception_messages = capture_exception_messages
        self.raise_error = raise_error

    @_safe_callback
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        framework_metadata = metadata or {}
        kind, name = self._chain_identity(serialized, framework_metadata, parent_run_id, kwargs)
        attributes = self._start_attributes(run_id, parent_run_id, framework_metadata)
        attributes["input"] = self._project_payload(inputs)
        self._start(run_id, parent_run_id, name, kind, attributes, framework_metadata)

    @_safe_callback
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._end(run_id, attributes={"output": self._project_payload(outputs)})

    @_safe_callback
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._error(run_id, error)

    @_safe_callback
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        model = self._model_name(serialized, kwargs)
        attributes = self._start_attributes(run_id, parent_run_id, metadata)
        attributes["model"] = model
        attributes["messages"] = self._project_payload(messages)
        self._start(
            run_id,
            parent_run_id,
            self._label("LLM", model),
            SpanKind.LLM,
            attributes,
            metadata,
        )

    @_safe_callback
    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: Any = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._capture_streaming_tokens:
            return None
        attributes = {"token": self._capture.capture(token, redaction=self._redaction)}
        if chunk is not None:
            attributes["chunk"] = self._project_payload(chunk)
        return self._invoke(
            "record streaming token",
            self._recorder.add_event,
            self._correlation_id(run_id),
            "llm.token",
            attributes=attributes,
        )

    @_safe_callback
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        usage = self._usage.extract(response)
        values = usage.to_attributes()
        if any(values.values()):
            self._invoke(
                "record token usage",
                self._recorder.record_metric,
                self._correlation_id(run_id),
                "llm.token.usage",
                values,
                cumulative=True,
            )
        self._end(
            run_id,
            attributes={
                "output": self._project_llm_result(response),
                "usage": values,
            },
        )

    @_safe_callback
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._error(run_id, error)

    @_safe_callback
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = self._serialized_name(serialized, "tool")
        attributes = self._start_attributes(run_id, parent_run_id, metadata)
        attributes["input"] = self._project_payload(inputs if inputs is not None else input_str)
        tool_call_id = kwargs.get("tool_call_id")
        normalized_tool_call_id = self._safe_optional_text(tool_call_id)
        if normalized_tool_call_id is not None:
            attributes["tool_call_id"] = normalized_tool_call_id
        self._start(
            run_id,
            parent_run_id,
            self._label("Tool", name),
            SpanKind.TOOL,
            attributes,
            metadata,
        )

    @_safe_callback
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        attributes: dict[str, object] = {"output": self._project_payload(output)}
        tool_call_id = kwargs.get("tool_call_id")
        if tool_call_id is None and isinstance(output, BaseMessage):
            tool_call_id = getattr(output, "tool_call_id", None)
        normalized_tool_call_id = self._safe_optional_text(tool_call_id)
        if normalized_tool_call_id is not None:
            attributes["tool_call_id"] = normalized_tool_call_id
        self._end(run_id, attributes=attributes)

    @_safe_callback
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._error(run_id, error)

    def _start(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        kind: SpanKind,
        attributes: dict[str, object],
        metadata: Mapping[str, object] | None,
    ) -> None:
        correlation_id = self._correlation_id(run_id)
        parent_id = self._optional_correlation_id(parent_run_id)
        lane_id = None
        lane_name = None
        if kind is SpanKind.NODE and metadata is not None:
            node = metadata.get("langgraph_node")
            if node is not None:
                lane_name = self._safe_text(node, "node")
                lane_id = self._safe_text(f"langgraph.node:{lane_name}", "langgraph.node")
        if parent_id is None:
            self._invoke(
                "start span",
                self._recorder.start_span,
                correlation_id,
                name,
                kind,
                parent_correlation_id=None,
                attributes=attributes,
                lane_id=lane_id,
                lane_name=lane_name,
                trace_name=name,
            )
        else:
            self._invoke(
                "start span",
                self._recorder.start_span,
                correlation_id,
                name,
                kind,
                parent_correlation_id=parent_id,
                attributes=attributes,
                lane_id=lane_id,
                lane_name=lane_name,
            )

    def _end(self, run_id: UUID, *, attributes: dict[str, object]) -> None:
        self._invoke(
            "end span",
            self._recorder.end_span,
            self._correlation_id(run_id),
            attributes=attributes,
        )

    def _error(self, run_id: UUID, error: BaseException) -> None:
        try:
            status = (
                SpanStatus.CANCELLED
                if isinstance(error, (asyncio.CancelledError, FuturesCancelledError))
                else SpanStatus.ERROR
            )
            error_type = self._exception_type(error)
            error_attributes: dict[str, object] = {"type": error_type}
            message = self._exception_message(error) if self._capture_exception_messages else None
            if message is not None:
                captured = self._capture.capture(
                    {"exception.message": message}, redaction=self._redaction
                )
                error_attributes["details"] = captured
            self._invoke(
                "end failed span",
                self._recorder.end_span,
                self._correlation_id(run_id),
                status=status,
                status_description=error_type,
                attributes={"error": error_attributes},
                preserve_application_error=True,
            )
        except BaseException:
            return

    def _start_attributes(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {"langchain.run_id": self._correlation_id(run_id)}
        parent_id = self._optional_correlation_id(parent_run_id)
        if parent_id is not None:
            attributes["langchain.parent_run_id"] = parent_id
        if metadata:
            projected = self._metadata.project(metadata)
            if projected:
                attributes["langgraph.metadata"] = projected
        return attributes

    def _chain_identity(
        self,
        serialized: Mapping[str, object] | None,
        metadata: Mapping[str, object],
        parent_run_id: UUID | None,
        kwargs: Mapping[str, object],
    ) -> tuple[SpanKind, str]:
        node = metadata.get("langgraph_node")
        if node is not None:
            return SpanKind.NODE, self._label("Node", self._safe_text(node, "node"))
        supplied_name = kwargs.get("name")
        graph_name = self._safe_optional_text(metadata.get("graph_name"))
        if graph_name is None:
            graph_name = self._safe_optional_text(supplied_name)
        if metadata.get("ls_integration") == "langgraph" and parent_run_id is None:
            return SpanKind.GRAPH, self._label("Graph", graph_name or "LangGraph")
        name = self._safe_optional_text(supplied_name) or self._serialized_name(serialized, "chain")
        if metadata.get("graph_name") is not None:
            return SpanKind.GRAPH, self._label("Graph", graph_name or "LangGraph")
        return SpanKind.CHAIN, name

    def _serialized_name(self, serialized: Mapping[str, object] | None, fallback: str) -> str:
        if serialized:
            name = serialized.get("name")
            normalized = self._safe_optional_text(name)
            if normalized is not None:
                return normalized
            identifier = serialized.get("id")
            if (
                isinstance(identifier, Sequence)
                and not isinstance(identifier, (str, bytes))
                and identifier
            ):
                normalized = self._safe_optional_text(identifier[-1])
                if normalized is not None:
                    return normalized
        return fallback

    def _model_name(
        self,
        serialized: Mapping[str, object] | None,
        kwargs: Mapping[str, object],
    ) -> str:
        invocation = kwargs.get("invocation_params")
        if isinstance(invocation, Mapping):
            for key in ("model", "model_name"):
                value = self._safe_optional_text(invocation.get(key))
                if value is not None:
                    return value
        return self._serialized_name(serialized, "model")

    def _project_payload(self, value: object) -> object:
        return self._capture.capture(
            value, redaction=self._redaction, normalize=normalize_framework_value
        )

    def _safe_optional_text(self, value: object) -> str | None:
        if type(value) is str:
            normalized = value.strip()
        elif type(value) in (int, float, bool):
            normalized = str(value)
        else:
            return None
        if not normalized:
            return None
        return normalized[: self._capture.max_string_length]

    def _safe_text(self, value: object, fallback: str) -> str:
        return self._safe_optional_text(value) or fallback[: self._capture.max_string_length]

    def _label(self, prefix: str, value: object) -> str:
        label = f"{prefix}: {self._safe_text(value, prefix.casefold())}"
        return label[: self._capture.max_string_length]

    def _exception_type(self, error: BaseException) -> str:
        try:
            error_type = type(error)
            module = error_type.__module__
            name = error_type.__qualname__
            if type(module) is str and type(name) is str:
                return self._safe_text(f"{module}.{name}", "BaseException")
        except BaseException:
            pass
        return self._safe_text("BaseException", "error")

    def _exception_message(self, error: BaseException) -> str | None:
        try:
            args = error.args
        except BaseException:
            return None
        if type(args) is not tuple:
            return None
        # Never join an unbounded exception argument collection or call __str__.
        if not args or type(args[0]) is not str:
            return None
        return self._safe_optional_text(args[0])

    def _project_llm_result(self, response: LLMResult) -> object:
        return self._project_payload(
            {
                "generations": response.generations,
                "llm_output": response.llm_output,
            }
        )

    def _invoke(
        self,
        operation: str,
        callback: Any,
        *args: object,
        preserve_application_error: bool = False,
        **kwargs: object,
    ) -> object | None:
        if preserve_application_error:
            try:
                return callback(*args, **kwargs)
            except BaseException:
                return None
        try:
            return callback(*args, **kwargs)
        except Exception:
            _LOGGER.warning("LangGraph callback adapter failed to %s", operation)
            if self.raise_error:
                raise
            return None

    @staticmethod
    def _correlation_id(run_id: UUID) -> str:
        if not isinstance(run_id, UUID):
            raise TypeError("run_id must be a UUID")
        return str(run_id)

    @classmethod
    def _optional_correlation_id(cls, run_id: UUID | None) -> str | None:
        return cls._correlation_id(run_id) if run_id is not None else None


__all__ = ["LangGraphCallbackHandler"]
