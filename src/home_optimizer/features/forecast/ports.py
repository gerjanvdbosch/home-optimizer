from __future__ import annotations

from typing import Any, Protocol

from home_optimizer.domain.forecast import ForecastEntry


class ForecastRepositoryPort(Protocol):
    source: str

    def write_entries(self, entries: list[ForecastEntry]) -> None: ...


class OpenMeteoGatewayPort(Protocol):
    def fetch_minutely_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        variables: list[str],
        forecast_steps: int,
        tilt: float | None = None,
        azimuth: float | None = None,
    ) -> dict[str, Any]: ...
