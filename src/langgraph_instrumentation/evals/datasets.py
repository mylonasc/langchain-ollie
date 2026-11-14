"""Portable JSON, JSONL and CSV evaluation datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ..models import JSONValue, attributes_to_dict
from .evaluators import (
    ContainsText,
    ExactText,
    MaxMetric,
    MinMetric,
    RegexText,
    ToolCalled,
)
from .models import EvalCase, EvalTurn

_SCHEMA_VERSION = "1"
_DATASET_KIND = "langgraph_instrumentation.eval_dataset"
_HEADER_KIND = "langgraph_instrumentation.eval_dataset_header"
_CSV_FIELDS = (
    "schema_version",
    "case_id",
    "turn_index",
    "role",
    "content",
    "messages_json",
    "input_json",
    "checks_json",
    "metadata_json",
    "tags",
    "timeout_seconds",
    "case_checks_json",
    "case_metadata_json",
    "case_tags",
    "name",
    "description",
)


def save_cases(cases: Sequence[EvalCase], path: str | Path) -> None:
    """Write cases in a format selected by the path suffix."""
    destination = Path(path)
    suffix = destination.suffix.casefold()
    if suffix == ".json":
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "kind": _DATASET_KIND,
            "cases": [_case_to_dict(case) for case in cases],
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if suffix == ".jsonl":
        lines = [
            json.dumps(
                {"schema_version": _SCHEMA_VERSION, "kind": _HEADER_KIND},
                sort_keys=True,
            )
        ]
        lines.extend(json.dumps(_case_to_dict(case), sort_keys=True) for case in cases)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if suffix == ".csv":
        _save_csv(cases, destination)
        return
    raise ValueError("dataset path must end with .json, .jsonl or .csv")


def load_cases(path: str | Path) -> tuple[EvalCase, ...]:
    """Load a versioned dataset from JSON, JSONL or CSV."""
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return _load_json(payload)
    if suffix == ".jsonl":
        return _load_jsonl(source.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return _load_csv(source)
    raise ValueError("dataset path must end with .json, .jsonl or .csv")


def _load_json(payload: object) -> tuple[EvalCase, ...]:
    if not isinstance(payload, Mapping):
        raise TypeError("dataset root must be an object")
    if payload.get("kind") != _DATASET_KIND:
        raise ValueError("not an evaluation dataset")
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation dataset version: {version!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise TypeError("cases must be a list")
    return _unique_cases(_case_from_dict(_mapping(item, "case")) for item in cases)


def _load_jsonl(text: str) -> tuple[EvalCase, ...]:
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("dataset is empty")
    first = records[0]
    if not isinstance(first, Mapping):
        raise TypeError("dataset records must be objects")
    if first.get("kind") == _HEADER_KIND:
        if first.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evaluation dataset version: {first.get('schema_version')!r}"
            )
        records = records[1:]
    return _unique_cases(_case_from_dict(_mapping(item, "case")) for item in records)


def _save_csv(cases: Sequence[EvalCase], destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for case in cases:
            case_checks = json.dumps(_checks_to_specs(case.checks), sort_keys=True)
            case_metadata = json.dumps(_json_mapping(case.metadata), sort_keys=True)
            for index, turn in enumerate(case.turns):
                writer.writerow(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "case_id": case.case_id,
                        "turn_index": index,
                        "role": "" if turn.input is not None else turn.messages[0][0],
                        "content": "" if turn.input is not None else turn.messages[0][1],
                        "messages_json": (
                            ""
                            if turn.input is not None
                            else json.dumps([list(item) for item in turn.messages])
                        ),
                        "input_json": (
                            ""
                            if turn.input is None
                            else json.dumps(_json_mapping(turn.input, "input"), sort_keys=True)
                        ),
                        "checks_json": json.dumps(_checks_to_specs(turn.checks), sort_keys=True),
                        "metadata_json": json.dumps(_json_mapping(turn.metadata), sort_keys=True),
                        "tags": ",".join(sorted(turn.tags)),
                        "timeout_seconds": (
                            "" if turn.timeout_seconds is None else turn.timeout_seconds
                        ),
                        "case_checks_json": case_checks,
                        "case_metadata_json": case_metadata,
                        "case_tags": ",".join(sorted(case.tags)),
                        "name": case.name or "",
                        "description": case.description or "",
                    }
                )


def _load_csv(source: Path) -> tuple[EvalCase, ...]:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dataset is empty")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        version = row.get("schema_version")
        if version not in (None, "", _SCHEMA_VERSION):
            raise ValueError(f"unsupported evaluation dataset version: {version!r}")
        case_id = row.get("case_id", "")
        if not case_id:
            raise ValueError("CSV rows must have a case_id")
        grouped.setdefault(case_id, []).append(row)

    cases: list[EvalCase] = []
    for case_id, case_rows in grouped.items():
        case_rows.sort(key=lambda row: int(row.get("turn_index") or 0))
        first = case_rows[0]
        turns: list[EvalTurn] = []
        for row in case_rows:
            input_text = (row.get("input_json") or "").strip()
            checks = _checks_from_specs(json.loads(row.get("checks_json") or "[]"))
            metadata = _mapping(json.loads(row.get("metadata_json") or "{}"), "metadata")
            tags = frozenset(tag for tag in (row.get("tags") or "").split(",") if tag)
            timeout = _csv_timeout(row.get("timeout_seconds"))
            if input_text:
                raw_input = json.loads(input_text)
                turns.append(
                    EvalTurn(
                        input=_json_mapping(raw_input, "input"),
                        checks=checks,
                        metadata=metadata,
                        tags=tags,
                        timeout_seconds=timeout,
                    )
                )
                continue
            messages_text = (row.get("messages_json") or "").strip()
            if messages_text:
                raw_messages = json.loads(messages_text)
                if not isinstance(raw_messages, list):
                    raise TypeError("messages_json must be a list")
                messages = tuple(
                    (str(item[0]), str(item[1]))
                    for item in raw_messages
                    if isinstance(item, list) and len(item) == 2
                )
            else:
                role = row.get("role") or "user"
                content = row.get("content") or ""
                messages = ((role, content),)
            turns.append(
                EvalTurn(
                    messages=messages,
                    checks=checks,
                    metadata=metadata,
                    tags=tags,
                    timeout_seconds=timeout,
                )
            )
        cases.append(
            EvalCase(
                case_id=case_id,
                turns=tuple(turns),
                checks=_checks_from_specs(json.loads(first.get("case_checks_json") or "[]")),
                name=(first.get("name") or None),
                description=(first.get("description") or None),
                metadata=_json_mapping(
                    json.loads(first.get("case_metadata_json") or "{}"), "metadata"
                ),
                tags=frozenset(tag for tag in (first.get("case_tags") or "").split(",") if tag),
            )
        )
    return tuple(cases)


def _case_to_dict(case: EvalCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "name": case.name,
        "description": case.description,
        "tags": sorted(case.tags),
        "metadata": _json_mapping(case.metadata),
        "turns": [_turn_to_dict(turn) for turn in case.turns],
        "checks": _checks_to_specs(case.checks),
    }


def _turn_to_dict(turn: EvalTurn) -> dict[str, object]:
    return {
        "input": None if turn.input is None else _json_mapping(turn.input, "input"),
        "messages": [[role, content] for role, content in turn.messages],
        "checks": _checks_to_specs(turn.checks),
        "metadata": _json_mapping(turn.metadata),
        "tags": sorted(turn.tags),
        "timeout_seconds": turn.timeout_seconds,
    }


def _case_from_dict(data: Mapping[str, object]) -> EvalCase:
    turns_data = data.get("turns")
    if not isinstance(turns_data, list) or not turns_data:
        raise ValueError("a case must define at least one turn")
    return EvalCase(
        case_id=_required_string(data, "case_id"),
        name=_optional_string(data, "name"),
        description=_optional_string(data, "description"),
        turns=tuple(_turn_from_dict(_mapping(item, "turn")) for item in turns_data),
        checks=_checks_from_specs(data.get("checks", [])),
        metadata=_json_mapping(data.get("metadata", {}), "metadata"),
        tags=frozenset(str(tag) for tag in _list(data.get("tags", []), "tags")),
    )


def _turn_from_dict(data: Mapping[str, object]) -> EvalTurn:
    input_value = data.get("input")
    messages_value = data.get("messages", [])
    if not isinstance(messages_value, list):
        raise TypeError("messages must be a list")
    messages: list[tuple[str, str]] = []
    for item in messages_value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise TypeError("messages must be [role, content] pairs")
        messages.append((item[0], item[1]))
    timeout = data.get("timeout_seconds")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))):
        raise TypeError("timeout_seconds must be a number or null")
    return EvalTurn(
        input=None if input_value is None else _json_mapping(input_value, "input"),
        messages=tuple(messages),
        checks=_checks_from_specs(data.get("checks", [])),
        metadata=_json_mapping(data.get("metadata", {}), "metadata"),
        tags=frozenset(str(tag) for tag in _list(data.get("tags", []), "tags")),
        timeout_seconds=timeout,
    )


def _checks_to_specs(checks: Sequence[object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for check in checks:
        spec: dict[str, object]
        if isinstance(check, ExactText):
            spec = {
                "type": "exact_text",
                "expected": check.expected,
                "case_sensitive": check.case_sensitive,
                "strip": check.strip,
            }
        elif isinstance(check, ContainsText):
            spec = {
                "type": "contains_text",
                "expected": check.expected,
                "case_sensitive": check.case_sensitive,
            }
        elif isinstance(check, RegexText):
            spec = {"type": "regex_text", "pattern": check.pattern, "flags": check.flags}
        elif isinstance(check, ToolCalled):
            spec = {
                "type": "tool_called",
                "tool": check.tool,
                "minimum": check.minimum,
            }
        elif isinstance(check, MaxMetric):
            spec = {
                "type": "max_metric",
                "metric": check.metric,
                "maximum": check.maximum,
                "missing": check.missing,
            }
        elif isinstance(check, MinMetric):
            spec = {
                "type": "min_metric",
                "metric": check.metric,
                "minimum": check.minimum,
                "missing": check.missing,
            }
        else:
            raise ValueError(
                "custom evaluators cannot be serialized to a portable dataset; "
                "attach them programmatically after loading"
            )
        specs.append(spec)
    return specs


def _checks_from_specs(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError("checks must be a list")
    checks: list[object] = []
    for item in value:
        spec = _mapping(item, "check")
        kind = _required_string(spec, "type")
        if kind == "exact_text":
            checks.append(
                ExactText(
                    _required_string(spec, "expected"),
                    case_sensitive=_bool_value(spec, "case_sensitive", True),
                    strip=_bool_value(spec, "strip", False),
                )
            )
        elif kind == "contains_text":
            checks.append(
                ContainsText(
                    _required_string(spec, "expected"),
                    case_sensitive=_bool_value(spec, "case_sensitive", True),
                )
            )
        elif kind == "regex_text":
            checks.append(
                RegexText(
                    _required_string(spec, "pattern"),
                    flags=_int_value(spec, "flags", 0),
                )
            )
        elif kind == "tool_called":
            checks.append(
                ToolCalled(
                    _required_string(spec, "tool"),
                    minimum=_int_value(spec, "minimum", 1),
                )
            )
        elif kind == "max_metric":
            checks.append(
                MaxMetric(
                    _required_string(spec, "metric"),
                    _required_number(spec, "maximum"),
                    missing=_optional_string(spec, "missing") or "fail",
                )
            )
        elif kind == "min_metric":
            checks.append(
                MinMetric(
                    _required_string(spec, "metric"),
                    _required_number(spec, "minimum"),
                    missing=_optional_string(spec, "missing") or "fail",
                )
            )
        else:
            raise ValueError(f"unsupported check type: {kind!r}")
    return tuple(checks)


def _unique_cases(cases: Iterable[EvalCase]) -> tuple[EvalCase, ...]:
    result = tuple(cases)
    identifiers: set[str] = set()
    for case in result:
        if case.case_id in identifiers:
            raise ValueError(f"duplicate case_id: {case.case_id!r}")
        identifiers.add(case.case_id)
    return result


def _json_mapping(value: object, field_name: str = "value") -> dict[str, JSONValue]:
    mapping = _mapping(value, field_name)
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return attributes_to_dict(mapping)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_number(data: Mapping[str, object], key: str) -> int | float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _int_value(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _bool_value(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _csv_timeout(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("timeout_seconds must be a number") from exc
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    return timeout
