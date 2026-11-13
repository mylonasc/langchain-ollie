"""Framework normalization, without deduplication or a second payload traversal."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation

from ..models import JSONValue
from ..policies import CaptureMode, CapturePolicy, RedactionPolicy


def normalize_framework_value(value: object) -> object:
    """Expose only known framework fields; never serialize arbitrary objects."""
    if isinstance(value, ChatGeneration):
        value = value.message
    if isinstance(value, BaseMessage):
        result: dict[str, object] = {"type": value.type, "content": value.content}
        for name in (
            "id",
            "name",
            "tool_call_id",
            "additional_kwargs",
            "response_metadata",
            "tool_calls",
        ):
            field = getattr(value, name, None)
            if field:
                result[name] = field
        return result
    if isinstance(value, Document):
        return {"page_content": value.page_content, "metadata": value.metadata, "id": value.id}
    if isinstance(value, Generation):
        return {"text": value.text, "generation_info": value.generation_info}
    return value


class MessageProjector:
    """Stateless message projection: every recording remains self-contained."""

    def __init__(
        self, *, capture: CapturePolicy | None = None, redaction: RedactionPolicy | None = None
    ) -> None:
        self._capture = capture or CapturePolicy()
        self._redaction = redaction or RedactionPolicy()

    def project(self, message: BaseMessage) -> JSONValue:
        if not isinstance(message, BaseMessage):
            raise TypeError("message must be a BaseMessage")
        if self._capture.mode is CaptureMode.NONE:
            return {"type": message.type}
        if self._capture.mode is CaptureMode.METADATA:
            return {"type": message.type, "content": self._capture.capture(message.content)}
        return self._capture.capture(
            message, redaction=self._redaction, normalize=normalize_framework_value
        )
