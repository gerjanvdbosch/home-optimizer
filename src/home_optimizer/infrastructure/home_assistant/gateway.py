from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import httpx
import websocket

from home_optimizer.domain.location import Location, parse_location


class HomeAssistantGateway:
    def __init__(
        self,
        base_url: str = "http://supervisor/core",
        websocket_url: str = "ws://supervisor/core/api/websocket",
        token: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.websocket_url = websocket_url.rstrip("/")
        self.token = token or os.getenv("SUPERVISOR_TOKEN")
        self.timeout = timeout
        if not self.token:
            raise ValueError("SUPERVISOR_TOKEN not found.")

        self.client = client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = self.client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        return response.json()

    def get_state(self, entity_id: str) -> dict[str, Any]:
        return self._get(f"/api/states/{entity_id}")

    def get_states(self) -> list[dict[str, Any]]:
        return self._get("/api/states")

    def get_location(self) -> Location | None:
        state = self.get_state("zone.home")
        attributes = state.get("attributes")
        if not isinstance(attributes, dict):
            return None

        return parse_location(attributes)

    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        minimal_response: bool = True,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"filter_entity_id": entity_id}
        if end_time:
            params["end_time"] = end_time.isoformat()
        if minimal_response:
            params["minimal_response"] = ""

        result = self._get(f"/api/history/period/{start_time.isoformat()}", params=params)
        if not result:
            return []
        return result[0]

    def get_statistics(
        self,
        statistic_id: str,
        start_time: datetime,
        end_time: datetime,
        period: str = "hour",
    ) -> list[dict[str, Any]]:
        ws = websocket.create_connection(self.websocket_url, timeout=self.timeout)
        try:
            _ws_recv(ws)  # auth_required
            ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth_msg = _ws_recv(ws)
            if auth_msg.get("type") != "auth_ok":
                raise RuntimeError(f"WebSocket auth failed: {auth_msg}")

            request_id = 1
            ws.send(json.dumps({
                "id": request_id,
                "type": "recorder/statistics_during_period",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "statistic_ids": [statistic_id],
                "period": period,
                "types": ["mean", "min", "max", "state", "sum"],
            }))
            result_msg = _ws_recv(ws)
            if not result_msg.get("success"):
                raise RuntimeError(f"statistics_during_period failed: {result_msg}")

            return result_msg.get("result", {}).get(statistic_id, [])
        finally:
            ws.close()


def _ws_recv(ws: websocket.WebSocket) -> dict[str, Any]:
    return json.loads(ws.recv())
