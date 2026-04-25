from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChartPoint:
    timestamp: str
    value: float


@dataclass(frozen=True)
class ChartSeries:
    name: str
    unit: str | None
    points: list[ChartPoint]
