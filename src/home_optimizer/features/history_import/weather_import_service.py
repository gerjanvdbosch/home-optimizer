from __future__ import annotations

import logging
from datetime import datetime, timedelta

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.location import Location
from home_optimizer.domain.time import ensure_utc
from home_optimizer.infrastructure.weather.openmeteo_entry_builder import (
    OpenMeteoForecastEntryBuilder,
)
from home_optimizer.infrastructure.weather.ports import (
    ForecastRepositoryPort,
    OpenMeteoGatewayPort,
)

LOGGER = logging.getLogger(__name__)


class WeatherImportService:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        location: Location | None,
        repository: ForecastRepositoryPort,
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
        self.builder = OpenMeteoForecastEntryBuilder(
            gateway,
            repository,
            pv_tilt=pv_tilt,
            pv_azimuth=pv_azimuth,
            living_room_window_azimuth=living_room_window_azimuth,
        )

    def import_weather_data(
        self,
        created_at: datetime | None = None,
    ) -> int:
        if self.location is None:
            LOGGER.info("Open-Meteo weather import skipped: home coordinates unavailable")
            return 0

        fetched_at = ensure_utc(created_at or utc_now())
        window_end = _floor_to_quarter_hour(fetched_at)
        past_days = self.history_days_back
        if past_days > 92:
            raise ValueError(
                "historische weerdata ligt verder terug dan Open-Meteo past_days ondersteunt"
            )

        entries = self.builder.build_entries(
            fetched_at=fetched_at,
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            forecast_steps=None,
            past_days=past_days,
            use_forecast_time_as_created_at=True,
        )
        window_start = window_end - timedelta(days=self.history_days_back)
        historical_entries = [
            entry
            for entry in entries
            if window_start <= ensure_utc(entry.forecast_time_utc) <= window_end
        ]
        inserted_rows = self.repository.write_new_entries(historical_entries)
        LOGGER.info("Stored %s historical Open-Meteo forecast values", inserted_rows)
        return inserted_rows


def _floor_to_quarter_hour(value: datetime) -> datetime:
    minute = (value.minute // 15) * 15
    return value.replace(minute=minute, second=0, microsecond=0)
