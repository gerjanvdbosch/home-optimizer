from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from home_optimizer.domain.pricing import (
    DEFAULT_CURRENCY,
    electricity_price_series,
    empty_electricity_price_series,
)
from home_optimizer.domain.series import NumericPoint, NumericSeries
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime


def _extract_price_points(payload: object, *, delivery_area: str) -> list[NumericPoint]:
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Nordpool response payload: {type(payload)}")

    entries = payload.get("multiAreaEntries", [])
    if not isinstance(entries, list):
        raise ValueError("Unexpected Nordpool response payload: multiAreaEntries must be a list")

    points: list[NumericPoint] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        price_by_area = entry.get("entryPerArea")
        if not isinstance(price_by_area, dict):
            continue

        price = price_by_area.get(delivery_area)
        if not isinstance(price, int | float):
            continue

        delivery_start = entry.get("deliveryStart")
        if not isinstance(delivery_start, str):
            continue

        points.append(
            NumericPoint(
                timestamp=normalize_utc_timestamp(parse_datetime(delivery_start)),
                value=float(price) / 1000.0,
            )
        )

    return points


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
        currency: str = DEFAULT_CURRENCY,
        market: str = "DayAhead",
    ) -> NumericSeries:
        params: dict[str, Any] = {
            "market": market,
            "deliveryArea": delivery_area,
            "currency": currency,
            "date": delivery_date.isoformat(),
        }
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return empty_electricity_price_series(currency)

        points = _extract_price_points(response.json(), delivery_area=delivery_area)

        return electricity_price_series(currency=currency, points=points)

