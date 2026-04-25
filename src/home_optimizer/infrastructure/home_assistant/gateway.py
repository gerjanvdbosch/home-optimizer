from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx


class HomeAssistantGateway:
    def __init__(
        self,
        base_url: str = "http://supervisor/core",
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv("SUPERVISOR_TOKEN")
        if not self.token:
            raise ValueError("SUPERVISOR_TOKEN not found.")

        self.client = httpx.Client(
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
