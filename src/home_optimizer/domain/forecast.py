from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ForecastEntry:
    created_at_utc: datetime
    forecast_time_utc: datetime
    name: str
    value: float
    source: str
