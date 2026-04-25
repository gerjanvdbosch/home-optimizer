from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from home_optimizer.bootstrap.settings import AppSettings
from home_optimizer.shared.sensors.definitions import SensorSpec
from home_optimizer.shared.time.clock import utc_now


@dataclass(frozen=True)
class HistoryImportRequest:
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


@dataclass(frozen=True)
class HistoryImportResult:
    imported_rows: dict[str, int]
