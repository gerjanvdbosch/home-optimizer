from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.sensors import SensorSpec


class HistoryImportRequest(DomainModel):
    specs: list[SensorSpec]
    start_time: datetime
    end_time: datetime

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        specs: list[SensorSpec],
    ) -> "HistoryImportRequest":
        end_time = utc_now()
        start_time = end_time - timedelta(days=settings.history_import_max_days_back)
        return cls(
            specs=specs,
            start_time=start_time,
            end_time=end_time,
        )


class HistoryImportResult(DomainModel):
    imported_rows: dict[str, int]
