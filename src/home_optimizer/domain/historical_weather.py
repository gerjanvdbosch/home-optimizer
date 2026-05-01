from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HistoricalWeatherEntry:
    timestamp_utc: datetime
    name: str
    value: float
    unit: str | None
    source: str
