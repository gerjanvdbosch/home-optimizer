from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from home_optimizer.domain.forecast import ForecastEntry


class ForecastRepositoryPort(Protocol):
    source: str

    def latest_created_at(self) -> datetime | None: ...

    def write_entries(self, entries: list[ForecastEntry]) -> None: ...

    def write_new_entries(self, entries: list[ForecastEntry]) -> int: ...


class OpenMeteoGatewayPort(Protocol):
    def fetch_minutely_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        variables: list[str],
        forecast_steps: int,
        past_days: int | None = None,
        tilt: float | None = None,
        azimuth: float | None = None,
    ) -> dict[str, Any]: ...
