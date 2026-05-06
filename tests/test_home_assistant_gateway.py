from __future__ import annotations

import httpx

from home_optimizer.domain.location import Location
from home_optimizer.infrastructure.home_assistant.gateway import HomeAssistantGateway


def test_home_assistant_gateway_reads_location_from_zone_home() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"attributes": {"latitude": "52.09", "longitude": 5.12}},
        )

    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert gateway.get_location() == Location(latitude=52.09, longitude=5.12)
    assert requested_urls == ["http://homeassistant.local/api/states/zone.home"]


def test_home_assistant_gateway_returns_none_for_missing_location() -> None:
    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    assert gateway.get_location() is None


def test_home_assistant_gateway_get_statistics_calls_correct_endpoint() -> None:
    requested: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append((str(request.url), request.content))
        return httpx.Response(
            200,
            json={"sensor.room": [{"start": "2026-04-14T00:00:00+00:00", "mean": 20.5}]},
        )

    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    from datetime import datetime, timezone

    start = datetime(2026, 4, 14, tzinfo=timezone.utc)
    end = datetime(2026, 4, 15, tzinfo=timezone.utc)
    result = gateway.get_statistics(statistic_id="sensor.room", start_time=start, end_time=end)

    assert len(result) == 1
    assert result[0]["mean"] == 20.5
    assert requested[0][0] == "http://homeassistant.local/api/recorder/statistics_during_period"


def test_home_assistant_gateway_get_statistics_returns_empty_for_unknown_entity() -> None:
    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        token="token",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
        ),
    )

    from datetime import datetime, timezone

    result = gateway.get_statistics(
        statistic_id="sensor.unknown",
        start_time=datetime(2026, 4, 14, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 15, tzinfo=timezone.utc),
    )

    assert result == []
