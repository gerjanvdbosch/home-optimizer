from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.sensors import SensorSpec


class HistoryImportRequest(DomainModel):
    specs: list[SensorSpec]
    start_time: datetime
    end_time: datetime


class HistoryImportResult(DomainModel):
    imported_rows: dict[str, int]
