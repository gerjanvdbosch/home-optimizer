from __future__ import annotations

from datetime import timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.features.history_import.schemas import HistoryImportRequest


def build_history_import_request(settings: AppSettings) -> HistoryImportRequest:
    end_time = utc_now()
    return HistoryImportRequest(
        specs=build_sensor_specs(settings),
        start_time=end_time - timedelta(days=settings.history_import_max_days_back),
        end_time=end_time,
    )

