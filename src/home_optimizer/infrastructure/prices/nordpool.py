from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from home_optimizer.domain.names import ELECTRICITY_PRICE
from home_optimizer.domain.series import NumericPoint, NumericSeries
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime


class NordpoolGateway:
    def __init__(
        self,
        base_url: str = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout))
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def fetch_day_ahead_prices(
        self,
        *,
        delivery_date: date,
        delivery_area: str,
        currency: str = "EUR",
        market: str = "DayAhead",
    ) -> NumericSeries:
        """Fetch day-ahead electricity prices for a CET calendar date.

        Returns a NumericSeries with one point per quarter-hour interval.
        The price unit is EUR/kWh (or the requested currency per kWh).
        Returns an empty series when no data is available for the requested date.
        """
        params: dict[str, Any] = {
            "market": market,
            "deliveryArea": delivery_area,
            "currency": currency,
            "date": delivery_date.isoformat(),
        }
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return NumericSeries(
                name=ELECTRICITY_PRICE,
                unit=f"{currency}/kWh",
                points=[],
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected Nordpool response payload: {type(payload)}")

        entries: list[dict[str, Any]] = payload.get("multiAreaEntries", [])
        points: list[NumericPoint] = []
        for entry in entries:
            price = entry.get("entryPerArea", {}).get(delivery_area)
            if price is None:
                continue
            delivery_start = entry.get("deliveryStart")
            if delivery_start is None:
                continue
            timestamp = normalize_utc_timestamp(parse_datetime(delivery_start))
            points.append(NumericPoint(timestamp=timestamp, value=float(price) / 1000.0))

        return NumericSeries(
            name=ELECTRICITY_PRICE,
            unit=f"{currency}/kWh",
            points=points,
        )
