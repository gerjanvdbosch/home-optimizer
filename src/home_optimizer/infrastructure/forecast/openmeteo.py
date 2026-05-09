from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_MAX_FORECAST_PAST_DAYS = 92


class OpenMeteoGateway:
    def __init__(
        self,
        base_url: str = OPEN_METEO_FORECAST_URL,
        historical_base_url: str = OPEN_METEO_HISTORICAL_FORECAST_URL,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.historical_base_url = historical_base_url
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout))
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

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
    ) -> dict[str, Any]:
        """Fetch 15-minute forecast data from Open-Meteo.

        When past_days exceeds the 92-day limit of the regular Forecast API,
        the Historical Forecast API is used automatically with equivalent
        start_date/end_date parameters.
        """
        use_historical = past_days is not None and past_days > _MAX_FORECAST_PAST_DAYS

        url = self.historical_base_url if use_historical else self.base_url
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": ",".join(variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }

        if use_historical:
            today = date.today()
            params["start_date"] = (today - timedelta(days=past_days)).isoformat()
            params["end_date"] = today.isoformat()
        else:
            if forecast_steps is not None:
                params["forecast_minutely_15"] = forecast_steps
            if past_days is not None:
                params["past_days"] = past_days

        if tilt is not None:
            params["tilt"] = tilt
        if azimuth is not None:
            params["azimuth"] = azimuth

        response = self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Open-Meteo response payload")
        return payload
