from __future__ import annotations

from datetime import date

import httpx
import pytest

from home_optimizer.domain.names import ELECTRICITY_PRICE
from home_optimizer.infrastructure.pricing.nordpool import NordpoolGateway


def _make_entry(delivery_start: str, delivery_end: str, price: float) -> dict:
    return {
        "deliveryStart": delivery_start,
        "deliveryEnd": delivery_end,
        "entryPerArea": {"NL": price},
    }


SAMPLE_PAYLOAD = {
    "deliveryDateCET": "2026-05-04",
    "version": 2,
    "market": "DayAhead",
    "currency": "EUR",
    "deliveryAreas": ["NL"],
    "multiAreaEntries": [
        _make_entry("2026-05-03T22:00:00Z", "2026-05-03T22:15:00Z", 126.31),
        _make_entry("2026-05-03T22:15:00Z", "2026-05-03T22:30:00Z", 120.09),
        _make_entry("2026-05-04T21:45:00Z", "2026-05-04T22:00:00Z", 98.50),
    ],
}


def test_nordpool_gateway_returns_numeric_series_with_correct_name_and_unit() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=SAMPLE_PAYLOAD))
    )
    gateway = NordpoolGateway(client=client)

    series = gateway.fetch_day_ahead_prices(
        delivery_date=date(2026, 5, 4),
        delivery_area="NL",
        currency="EUR",
    )

    assert series.name == ELECTRICITY_PRICE
    assert series.unit == "EUR/kWh"


def test_nordpool_gateway_maps_delivery_start_to_timestamp() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=SAMPLE_PAYLOAD))
    )
    gateway = NordpoolGateway(client=client)

    series = gateway.fetch_day_ahead_prices(
        delivery_date=date(2026, 5, 4),
        delivery_area="NL",
    )

    assert len(series.points) == 3
    assert series.points[0].timestamp == "2026-05-03T22:00:00+00:00"
    assert series.points[0].value == pytest.approx(126.31 / 1000)
    assert series.points[1].timestamp == "2026-05-03T22:15:00+00:00"
    assert series.points[1].value == pytest.approx(120.09 / 1000)
    assert series.points[2].timestamp == "2026-05-04T21:45:00+00:00"
    assert series.points[2].value == pytest.approx(98.50 / 1000)


def test_nordpool_gateway_returns_empty_series_on_204() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(204))
    )
    gateway = NordpoolGateway(client=client)

    series = gateway.fetch_day_ahead_prices(
        delivery_date=date(2026, 5, 4),
        delivery_area="NL",
    )

    assert series.name == ELECTRICITY_PRICE
    assert series.points == []


def test_nordpool_gateway_skips_entry_when_area_missing() -> None:
    payload = {
        "deliveryDateCET": "2026-05-04",
        "market": "DayAhead",
        "currency": "EUR",
        "deliveryAreas": ["NL"],
        "multiAreaEntries": [
            {"deliveryStart": "2026-05-03T22:00:00Z", "deliveryEnd": "2026-05-03T22:15:00Z", "entryPerArea": {}},
            _make_entry("2026-05-03T22:15:00Z", "2026-05-03T22:30:00Z", 115.0),
        ],
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    gateway = NordpoolGateway(client=client)

    series = gateway.fetch_day_ahead_prices(
        delivery_date=date(2026, 5, 4),
        delivery_area="NL",
    )

    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(115.0 / 1000)


def test_nordpool_gateway_raises_on_http_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    )
    gateway = NordpoolGateway(client=client)

    with pytest.raises(httpx.HTTPStatusError):
        gateway.fetch_day_ahead_prices(
            delivery_date=date(2026, 5, 4),
            delivery_area="NL",
        )


def test_nordpool_gateway_sends_correct_query_parameters() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=SAMPLE_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = NordpoolGateway(client=client)

    gateway.fetch_day_ahead_prices(
        delivery_date=date(2026, 5, 4),
        delivery_area="NL",
        currency="EUR",
        market="DayAhead",
    )

    assert len(seen) == 1
    url = seen[0]
    assert "market=DayAhead" in url
    assert "deliveryArea=NL" in url
    assert "currency=EUR" in url
    assert "date=2026-05-04" in url

