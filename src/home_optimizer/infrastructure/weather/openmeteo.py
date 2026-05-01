from __future__ import annotations

from typing import Any

import httpx


class OpenMeteoGateway:
    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
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
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": ",".join(variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
        if forecast_steps is not None:
            params["forecast_minutely_15"] = forecast_steps
        if past_days is not None:
            params["past_days"] = past_days
        if tilt is not None:
            params["tilt"] = tilt
        if azimuth is not None:
            params["azimuth"] = azimuth

        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Open-Meteo response payload")
        return payload
