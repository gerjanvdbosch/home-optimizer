from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from home_optimizer.bootstrap.settings import AppSettings
from home_optimizer.shared.sensors.definitions import SensorSpec
from home_optimizer.shared.time.clock import utc_now
from home_optimizer.shared.time.parse import parse_datetime


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
        end_time = (
            parse_datetime(settings.history_import_end)
            if settings.history_import_end
            else utc_now()
        )
        return cls(
            specs=specs,
            start_time=parse_datetime(settings.history_import_start),
            end_time=end_time,
        )


@dataclass(frozen=True)
class HistoryImportResult:
    imported_rows: dict[str, int]
