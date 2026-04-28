from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericPoint:
    timestamp: str
    value: float


@dataclass(frozen=True)
class NumericSeries:
    name: str
    unit: str | None
    points: list[NumericPoint]


@dataclass(frozen=True)
class TextPoint:
    timestamp: str
    value: str


@dataclass(frozen=True)
class TextSeries:
    name: str
    points: list[TextPoint]
