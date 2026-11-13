"""Finite, lifetime admission budgets for a local recording."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecordingLimits:
    """Start a new recorder to replenish these budgets; completion is never denied."""

    max_spans: int = 4_096
    max_events: int = 8_192
    max_metrics: int = 4_096
    max_lanes: int = 256
    max_attribute_bytes: int = 16_384

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
