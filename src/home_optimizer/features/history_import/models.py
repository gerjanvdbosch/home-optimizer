from __future__ import annotations

from datetime import datetime
from typing import Any

from home_optimizer.domain.models import DomainModel


class SensorPoint(DomainModel):
    timestamp: datetime
    value: Any


class MinuteSample(DomainModel):
    timestamp_minute: datetime
    name: str
    source: str
    entity_id: str
    category: str | None
    unit: str | None
    mean_real: float | None = None
    min_real: float | None = None
    max_real: float | None = None
    last_real: float | None = None
    last_text: str | None = None
    last_bool: int | None = None
    sample_count: int


class ImportChunkWindow(DomainModel):
    start_time: datetime
    end_time: datetime

