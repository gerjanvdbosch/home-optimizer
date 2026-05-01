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


class OpenMeteoForecastService:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        location: Location | None,
        repository: ForecastRepositoryPort,
        *,
        pv_tilt: float | None,
        pv_azimuth: float | None,
        living_room_window_azimuth: float | None,
        poll_interval_seconds: int,
        forecast_steps: int = 192,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")

        self.location = location
        self.repository = repository
        self.poll_interval = timedelta(seconds=poll_interval_seconds)
        self.forecast_steps = forecast_steps
        self.builder = OpenMeteoForecastEntryBuilder(
            gateway,
            repository,
            pv_tilt=pv_tilt,
            pv_azimuth=pv_azimuth,
            living_room_window_azimuth=living_room_window_azimuth,
        )

    @property
    def enabled(self) -> bool:
        return True

    def refresh_forecast(self, created_at: datetime | None = None) -> int:
        fetched_at = ensure_utc(created_at or utc_now())
        latest_created_at = self.repository.latest_created_at()
        if latest_created_at is not None and fetched_at - latest_created_at < self.poll_interval:
            LOGGER.info(
                "Open-Meteo forecast refresh skipped: latest forecast is still fresh",
            )
            return 0

        if self.location is None:
            LOGGER.info("Open-Meteo forecast refresh skipped: home coordinates unavailable")
            return 0

        entries = self.builder.build_entries(
            fetched_at=fetched_at,
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            forecast_steps=self.forecast_steps,
        )
        self.repository.write_entries(entries)
        LOGGER.info("Stored %s Open-Meteo forecast values", len(entries))
        return len(entries)
