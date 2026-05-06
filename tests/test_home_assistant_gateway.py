from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
        websocket_url="ws://homeassistant.local/api/websocket",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert gateway.get_location() == Location(latitude=52.09, longitude=5.12)
    assert requested_urls == ["http://homeassistant.local/api/states/zone.home"]


def test_home_assistant_gateway_returns_none_for_missing_location() -> None:
    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        websocket_url="ws://homeassistant.local/api/websocket",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    assert gateway.get_location() is None


def test_home_assistant_gateway_get_statistics_calls_correct_endpoint() -> None:
    messages = [
        json.dumps({"type": "auth_required"}),
        json.dumps({"type": "auth_ok"}),
        json.dumps({
            "id": 1,
            "type": "result",
            "success": True,
            "result": {"sensor.room": [{"start": "2026-04-14T00:00:00+00:00", "mean": 20.5}]},
        }),
    ]
    sent: list[str] = []

    mock_ws = MagicMock()
    mock_ws.recv.side_effect = messages
    mock_ws.send.side_effect = lambda msg: sent.append(msg)

    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        websocket_url="ws://homeassistant.local/api/websocket",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    with patch("home_optimizer.infrastructure.home_assistant.gateway.websocket.create_connection", return_value=mock_ws) as create_connection:
        result = gateway.get_statistics(
            statistic_id="sensor.room",
            start_time=datetime(2026, 4, 14, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 15, tzinfo=timezone.utc),
        )

    assert len(result) == 1
    assert result[0]["mean"] == 20.5
    create_connection.assert_called_once_with("ws://homeassistant.local/api/websocket", timeout=60.0)
    auth_payload = json.loads(sent[0])
    assert auth_payload == {"type": "auth", "access_token": "token"}
    cmd_payload = json.loads(sent[1])
    assert cmd_payload["type"] == "recorder/statistics_during_period"
    assert cmd_payload["statistic_ids"] == ["sensor.room"]
    mock_ws.close.assert_called_once()


def test_home_assistant_gateway_get_statistics_returns_empty_for_unknown_entity() -> None:
    messages = [
        json.dumps({"type": "auth_required"}),
        json.dumps({"type": "auth_ok"}),
        json.dumps({"id": 1, "type": "result", "success": True, "result": {}}),
    ]

    mock_ws = MagicMock()
    mock_ws.recv.side_effect = messages

    gateway = HomeAssistantGateway(
        base_url="http://homeassistant.local",
        websocket_url="ws://homeassistant.local/api/websocket",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    with patch("home_optimizer.infrastructure.home_assistant.gateway.websocket.create_connection", return_value=mock_ws):
        result = gateway.get_statistics(
            statistic_id="sensor.unknown",
            start_time=datetime(2026, 4, 14, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 15, tzinfo=timezone.utc),
        )

    assert result == []


def test_home_assistant_gateway_get_statistics_supports_secure_websocket_url() -> None:
    messages = [
        json.dumps({"type": "auth_required"}),
        json.dumps({"type": "auth_ok"}),
        json.dumps({"id": 1, "type": "result", "success": True, "result": {}}),
    ]

    mock_ws = MagicMock()
    mock_ws.recv.side_effect = messages

    gateway = HomeAssistantGateway(
        base_url="https://homeassistant.local/core",
        websocket_url="wss://homeassistant.local/core/api/websocket",
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))),
    )

    with patch("home_optimizer.infrastructure.home_assistant.gateway.websocket.create_connection", return_value=mock_ws) as create_connection:
        gateway.get_statistics(
            statistic_id="sensor.room",
            start_time=datetime(2026, 4, 14, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 15, tzinfo=timezone.utc),
        )

    create_connection.assert_called_once_with("wss://homeassistant.local/core/api/websocket", timeout=60.0)

