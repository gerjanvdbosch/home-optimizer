from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.historical_weather import HistoricalWeatherEntry


class ForecastRepositoryPort(Protocol):
    source: str

    def latest_created_at(self) -> datetime | None: ...

    def write_entries(self, entries: list[ForecastEntry]) -> None: ...

    def write_new_entries(self, entries: list[ForecastEntry]) -> int: ...


class HistoricalWeatherRepositoryPort(Protocol):
    source: str

    def write_new_entries(self, entries: list[HistoricalWeatherEntry]) -> int: ...


class OpenMeteoGatewayPort(Protocol):
    def fetch_minutely_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        variables: list[str],
        forecast_steps: int | None = None,
        past_days: int | None = None,
        tilt: float | None = None,
        azimuth: float | None = None,
    ) -> dict[str, Any]: ...

    def fetch_hourly_historical_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        variables: list[str],
        start_date: date,
        end_date: date,
        tilt: float | None = None,
        azimuth: float | None = None,
    ) -> dict[str, Any]: ...
