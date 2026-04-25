from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.history_import.schemas import HistoryImportRequest


def build_history_import_request(
    settings: AppSettings,
    now: datetime | None = None,
) -> HistoryImportRequest:
    current_time = ensure_utc(now or utc_now())
    return HistoryImportRequest(
        specs=build_sensor_specs(settings),
        start_time=current_time - timedelta(days=settings.history_import_max_days_back),
        end_time=current_time,
    )
