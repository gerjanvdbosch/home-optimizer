from __future__ import annotations

import logging
from datetime import datetime, timedelta

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.location import Location
from home_optimizer.domain.time import ensure_utc
from home_optimizer.infrastructure.weather.openmeteo_historical_weather_builder import (
    OpenMeteoHistoricalWeatherBuilder,
)
from home_optimizer.infrastructure.weather.ports import (
    HistoricalWeatherRepositoryPort,
    OpenMeteoGatewayPort,
)

LOGGER = logging.getLogger(__name__)


class HistoricalWeatherImportService:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        location: Location | None,
        repository: HistoricalWeatherRepositoryPort,
        *,
        pv_tilt: float | None,
        pv_azimuth: float | None,
        living_room_window_azimuth: float | None,
        history_days_back: int,
    ) -> None:
        if history_days_back <= 0:
            raise ValueError("history_days_back must be greater than zero")

        self.location = location
        self.repository = repository
        self.history_days_back = history_days_back
        self.builder = OpenMeteoHistoricalWeatherBuilder(
            gateway,
            repository,
            pv_tilt=pv_tilt,
            pv_azimuth=pv_azimuth,
            living_room_window_azimuth=living_room_window_azimuth,
        )

    def import_historical_weather(
        self,
        created_at: datetime | None = None,
    ) -> int:
        if self.location is None:
            LOGGER.info("Historical weather import skipped: home coordinates unavailable")
            return 0

        fetched_at = ensure_utc(created_at or utc_now())
        window_end = _floor_to_hour(fetched_at)
        window_start = window_end - timedelta(days=self.history_days_back)
        entries = self.builder.build_entries(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            start_date=window_start.date(),
            end_date=window_end.date(),
        )
        historical_entries = [
            entry
            for entry in entries
            if window_start <= ensure_utc(entry.timestamp_utc) <= window_end
        ]
        inserted_rows = self.repository.write_new_entries(historical_entries)
        LOGGER.info("Stored %s historical weather values", inserted_rows)
        return inserted_rows


def _floor_to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)

