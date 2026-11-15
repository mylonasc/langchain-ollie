"""Safe capture and normalization policies for instrumentation payloads."""

from __future__ import annotations

import json
import math
from base64 import b64encode
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import islice

from .models import JSONValue

type RedactionPredicate = Callable[[tuple[str, ...], str, object], bool]

_DEFAULT_REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    }
)
_DEFAULT_METADATA_KEYS = frozenset(
    {
        "agent_name",
        "checkpoint_id",
        "graph_name",
        "langgraph_node",
        "thread_id",
    }
)


class CaptureMode(StrEnum):
    """Amount of application payload retained by instrumentation."""

    NONE = "none"
    METADATA = "metadata"
    BOUNDED = "bounded"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Determines which mapping values are replaced before capture."""

    keys: frozenset[str] = _DEFAULT_REDACT_KEYS
    replacement: str = "<redacted>"
    predicate: RedactionPredicate | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = frozenset(key.casefold() for key in self.keys)
        if any(not key for key in normalized):
            raise ValueError("redaction keys must not be empty")
        if not self.replacement:
            raise ValueError("redaction replacement must not be empty")
        object.__setattr__(self, "keys", normalized)

    def should_redact(self, path: tuple[str, ...], key: str, value: object) -> bool:
        if key.casefold() in self.keys:
            return True
        return self.predicate(path, key, value) if self.predicate is not None else False


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    """Configures how much of a payload is captured."""

    mode: CaptureMode = CaptureMode.METADATA
    max_depth: int = 6
    max_string_length: int = 2_000
    max_collection_items: int = 50
    max_byte_length: int = 0
    max_nodes: int = 2_048
    max_payload_bytes: int = 16_384

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CaptureMode):
            raise TypeError("mode must be a CaptureMode")
        _validate_positive(self.max_depth, "max_depth")
        _validate_positive(self.max_string_length, "max_string_length")
        _validate_positive(self.max_collection_items, "max_collection_items")
        _validate_non_negative(self.max_byte_length, "max_byte_length")
        _validate_positive(self.max_nodes, "max_nodes")
        _validate_positive(self.max_payload_bytes, "max_payload_bytes")

    @classmethod
    def none(cls) -> CapturePolicy:
        return cls(mode=CaptureMode.NONE)

    @classmethod
    def metadata(cls, *, max_collection_items: int = 50) -> CapturePolicy:
        return cls(
            mode=CaptureMode.METADATA,
            max_collection_items=max_collection_items,
        )

    @classmethod
    def bounded(
        cls,
        *,
        max_depth: int = 6,
        max_string_length: int = 2_000,
        max_collection_items: int = 50,
        max_byte_length: int = 0,
    ) -> CapturePolicy:
        return cls(
            mode=CaptureMode.BOUNDED,
            max_depth=max_depth,
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
            max_byte_length=max_byte_length,
        )

    @classmethod
    def full(cls) -> CapturePolicy:
        return cls(mode=CaptureMode.FULL)

    def capture(
        self,
        value: object,
        *,
        redaction: RedactionPolicy | None = None,
        normalize: Callable[[object], object] | None = None,
    ) -> JSONValue:
        if self.mode is CaptureMode.NONE:
            return None
        if self.mode is CaptureMode.METADATA:
            return _metadata_summary(value)
        sanitizer = Sanitizer(
            redaction=redaction,
            max_depth=self.max_depth if self.mode is CaptureMode.BOUNDED else 64,
            max_string_length=(
                self.max_string_length if self.mode is CaptureMode.BOUNDED else None
            ),
            max_collection_items=(
                self.max_collection_items if self.mode is CaptureMode.BOUNDED else None
            ),
            max_byte_length=self.max_byte_length if self.mode is CaptureMode.BOUNDED else None,
            max_nodes=self.max_nodes,
            max_payload_bytes=self.max_payload_bytes,
            normalize=normalize,
        )
        return sanitizer.sanitize(value)


class Sanitizer:
    """Converts arbitrary values to deterministic, redacted JSON values."""

    def __init__(
        self,
        *,
        redaction: RedactionPolicy | None = None,
        max_depth: int | None = 6,
        max_string_length: int | None = 2_000,
        max_collection_items: int | None = 50,
        max_byte_length: int | None = 0,
        max_nodes: int = 2_048,
        max_payload_bytes: int = 16_384,
        normalize: Callable[[object], object] | None = None,
    ) -> None:
        _validate_optional_positive(max_depth, "max_depth")
        _validate_optional_positive(max_string_length, "max_string_length")
        _validate_optional_positive(max_collection_items, "max_collection_items")
        _validate_optional_non_negative(max_byte_length, "max_byte_length")
        _validate_positive(max_nodes, "max_nodes")
        _validate_positive(max_payload_bytes, "max_payload_bytes")
        self._redaction = redaction or RedactionPolicy()
        self._max_depth = min(max_depth or 64, 64)
        self._max_string_length = max_string_length
        self._max_collection_items = max_collection_items
        self._max_byte_length = max_byte_length
        self._max_nodes = max_nodes
        self._max_payload_bytes = max_payload_bytes
        self._normalize = normalize

    def sanitize(self, value: object) -> JSONValue:
        result = self._sanitize(value, path=(), depth=0, active_ids=set(), budget=[self._max_nodes])
        # The traversal budget also bounds intermediate objects. The final encoded
        # byte budget includes keys/markers/escaping, not just string characters.
        if len(json.dumps(result, ensure_ascii=True).encode()) > self._max_payload_bytes:
            return "<payload-budget>"
        return result

    def _sanitize(
        self,
        value: object,
        *,
        path: tuple[str, ...],
        depth: int,
        active_ids: set[int],
        budget: list[int],
    ) -> JSONValue:
        if budget[0] <= 0:
            return "<node-budget>"
        budget[0] -= 1
        original = value
        if self._normalize is not None:
            value = self._normalize(value)
        if value is None or isinstance(value, (str, bool, int, float, bytes)):
            return self._sanitize_scalar(value)
        if self._max_depth is not None and depth >= self._max_depth:
            return "<max-depth>"

        value_id = id(original)
        if value_id in active_ids:
            return "<cycle>"
        active_ids.add(value_id)
        try:
            if isinstance(value, Mapping):
                return self._sanitize_mapping(value, path, depth, active_ids, budget)
            if isinstance(value, (list, tuple, set, frozenset)):
                return self._sanitize_collection(value, path, depth, active_ids, budget)
            return f"<unsupported:{type(value).__name__}>"
        finally:
            active_ids.remove(value_id)

    def _sanitize_scalar(self, value: str | bool | int | float | bytes | None) -> JSONValue:
        if isinstance(value, str):
            if len(value) > self._max_payload_bytes:
                return "<payload-budget>"
            if self._max_string_length is None or len(value) <= self._max_string_length:
                return value
            omitted = len(value) - self._max_string_length
            return f"{value[: self._max_string_length]}<truncated:{omitted}>"
        if isinstance(value, bytes):
            if self._max_byte_length is None and len(value) > self._max_payload_bytes:
                return {"type": "bytes", "length": len(value), "truncated": len(value)}
            captured = value[
                : min(
                    self._max_byte_length if self._max_byte_length is not None else len(value),
                    self._max_payload_bytes,
                )
            ]
            result: dict[str, JSONValue] = {
                "type": "bytes",
                "length": len(value),
            }
            if captured:
                result["base64"] = b64encode(captured).decode("ascii")
            if len(captured) < len(value):
                result["truncated"] = len(value) - len(captured)
            return result
        if isinstance(value, float) and not math.isfinite(value):
            return f"<non-finite:{value}>"
        if isinstance(value, int) and value.bit_length() > 256:
            return "<integer-budget>"
        return value

    def _sanitize_mapping(
        self,
        value: Mapping[object, object],
        path: tuple[str, ...],
        depth: int,
        active_ids: set[int],
        budget: list[int],
    ) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {}
        item_limit = self._max_collection_items
        items = iter(value.items())
        item_limit = min(item_limit or self._max_nodes, max(1, budget[0]))
        selected = list(islice(items, item_limit + 1))
        has_more = len(selected) > item_limit
        if has_more:
            selected.pop()
        for raw_key, item in selected:
            key = self._unique_key(self._safe_key(raw_key), result)
            redaction_key = raw_key if isinstance(raw_key, str) else key
            if self._redaction.should_redact(path, redaction_key, item):
                result[key] = self._redaction.replacement
            else:
                path_key = raw_key if isinstance(raw_key, str) else key
                result[key] = self._sanitize(
                    item,
                    path=(*path, path_key),
                    depth=depth + 1,
                    active_ids=active_ids,
                    budget=budget,
                )
        if has_more:
            marker = self._unique_key("<truncated>", result)
            result[marker] = max(1, len(value) - len(selected)) if type(value) is dict else 1
        return result

    def _sanitize_collection(
        self,
        value: list[object] | tuple[object, ...] | set[object] | frozenset[object],
        path: tuple[str, ...],
        depth: int,
        active_ids: set[int],
        budget: list[int],
    ) -> list[JSONValue]:
        if isinstance(value, (set, frozenset)):
            if len(value) > min(self._max_collection_items or self._max_nodes, budget[0]):
                return [f"<set-items-omitted:{len(value)}>"]
            sanitized = [
                self._sanitize(
                    item,
                    path=(*path, "set-item"),
                    depth=depth + 1,
                    active_ids=active_ids,
                    budget=budget,
                )
                for item in value
            ]
            return sorted(sanitized, key=lambda item: json.dumps(item, sort_keys=True))

        item_limit = self._max_collection_items
        item_limit = min(item_limit or self._max_nodes, max(1, budget[0]))
        selected = list(islice(iter(value), item_limit + 1))
        has_more = len(selected) > item_limit
        if has_more:
            selected.pop()
        result = [
            self._sanitize(
                item,
                path=(*path, str(index)),
                depth=depth + 1,
                active_ids=active_ids,
                budget=budget,
            )
            for index, item in enumerate(selected)
        ]
        if has_more:
            result.append(f"<truncated:{max(1, len(value) - len(selected))}>")
        return result

    def _safe_key(self, value: object) -> str:
        if not isinstance(value, str):
            return f"<key:{type(value).__name__}>"
        if len(value) > self._max_payload_bytes:
            return "<key-budget>"
        if self._max_string_length is None or len(value) <= self._max_string_length:
            return value
        omitted = len(value) - self._max_string_length
        return f"{value[: self._max_string_length]}<truncated:{omitted}>"

    @staticmethod
    def _unique_key(key: str, result: Mapping[str, JSONValue]) -> str:
        if key not in result:
            return key
        suffix = 2
        while f"{key}#{suffix}" in result:
            suffix += 1
        return f"{key}#{suffix}"


@dataclass(frozen=True, slots=True)
class MetadataProjector:
    """Selects and sanitizes framework metadata."""

    include_keys: frozenset[str] | None = _DEFAULT_METADATA_KEYS
    exclude_keys: frozenset[str] = frozenset()
    capture: CapturePolicy = field(default_factory=CapturePolicy.bounded)
    redaction: RedactionPolicy = field(default_factory=RedactionPolicy)

    def __post_init__(self) -> None:
        if self.include_keys is not None:
            object.__setattr__(self, "include_keys", frozenset(self.include_keys))
        object.__setattr__(self, "exclude_keys", frozenset(self.exclude_keys))

    def project(self, metadata: Mapping[str, object]) -> dict[str, JSONValue]:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        if self.include_keys is not None:
            selected = [
                (key, metadata[key])
                for key in sorted(self.include_keys)
                if key in metadata and key not in self.exclude_keys
            ]
            has_more = False
        else:
            item_limit = min(
                self.capture.max_nodes,
                self.capture.max_nodes
                if self.capture.mode is CaptureMode.FULL
                else self.capture.max_collection_items,
            )
            items = iter(metadata.items())
            selected = list(islice(items, item_limit + 1))
            has_more = len(selected) > item_limit
            if has_more:
                selected.pop()
            if any(not isinstance(key, str) for key, _ in selected):
                raise TypeError("metadata keys must be strings")
            selected.sort(key=lambda item: item[0])

        result: dict[str, JSONValue] = {}
        for key, value in selected:
            if key in self.exclude_keys:
                continue
            if self.redaction.should_redact((), key, value):
                result[key] = self.redaction.replacement
            else:
                result[key] = self.capture.capture(value, redaction=self.redaction)
        if has_more:
            marker = "<truncated>"
            while marker in result:
                marker += "#"
            result[marker] = max(1, len(metadata) - len(selected))
        return result


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token accounting from a model response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_attributes(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageExtractor:
    """Normalizes token usage across common LangChain provider response shapes."""

    _INPUT_KEYS = (
        "input_tokens",
        "prompt_tokens",
        "input_token_count",
        "prompt_token_count",
    )
    _OUTPUT_KEYS = (
        "output_tokens",
        "completion_tokens",
        "output_token_count",
        "candidates_token_count",
    )
    _TOTAL_KEYS = ("total_tokens", "total_token_count")

    def extract(self, response: object) -> TokenUsage:
        for value in (
            getattr(response, "llm_output", None),
            getattr(response, "usage_metadata", None),
            getattr(response, "response_metadata", None),
        ):
            usage = self._from_container(value)
            if usage is not None:
                return usage

        per_generation: list[TokenUsage] = []
        generations = getattr(response, "generations", None)
        if isinstance(generations, list):
            for generation_group in generations:
                group = (
                    generation_group if isinstance(generation_group, list) else [generation_group]
                )
                for generation in group:
                    message = getattr(generation, "message", None)
                    for value in (
                        getattr(generation, "usage_metadata", None),
                        getattr(generation, "response_metadata", None),
                        getattr(generation, "generation_info", None),
                        getattr(message, "usage_metadata", None),
                        getattr(message, "response_metadata", None),
                    ):
                        usage = self._from_container(value)
                        if usage is not None:
                            per_generation.append(usage)
                            break
        if not per_generation:
            return TokenUsage()
        return TokenUsage(
            input_tokens=sum(item.input_tokens for item in per_generation),
            output_tokens=sum(item.output_tokens for item in per_generation),
            total_tokens=sum(item.total_tokens for item in per_generation),
        )

    def _from_container(self, value: object) -> TokenUsage | None:
        if not isinstance(value, Mapping):
            return None
        for key in ("token_usage", "usage", "usage_metadata"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                usage = self._from_mapping(nested)
                if usage is not None:
                    return usage
        return self._from_mapping(value)

    def _from_mapping(self, usage: Mapping[str, object]) -> TokenUsage | None:
        recognized_keys = self._INPUT_KEYS + self._OUTPUT_KEYS + self._TOTAL_KEYS
        if not any(key in usage for key in recognized_keys):
            return None
        input_tokens = _first_token_count(usage, self._INPUT_KEYS)
        output_tokens = _first_token_count(usage, self._OUTPUT_KEYS)
        total_tokens = _first_token_count(usage, self._TOTAL_KEYS)
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return TokenUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=total_tokens,
        )


def _metadata_summary(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        return (
            {"type": "mapping", "size": len(value)} if type(value) is dict else {"type": "mapping"}
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"type": type(value).__name__, "size": len(value)}
    if isinstance(value, (bool, int, float)):
        return {"type": type(value).__name__}
    return {"type": type(value).__name__}


def _first_token_count(usage: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in usage:
            continue
        value = usage[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and len(value) <= 20 and value.isdecimal():
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _validate_positive(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _validate_optional_positive(value: int | None, field_name: str) -> None:
    if value is not None:
        _validate_positive(value, field_name)


def _validate_non_negative(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_optional_non_negative(value: int | None, field_name: str) -> None:
    if value is not None:
        _validate_non_negative(value, field_name)
