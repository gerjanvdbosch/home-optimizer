from __future__ import annotations

import httpx

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

    assert gateway.get_location() == (52.09, 5.12)
    assert requested_urls == ["http://homeassistant.local/api/states/zone.home"]


def test_home_assistant_gateway_returns_none_for_missing_location() -> None:
    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    assert gateway.get_location() is None
