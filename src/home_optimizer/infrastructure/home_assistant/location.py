from __future__ import annotations

from typing import Any, Protocol

HOME_ZONE_ENTITY_ID = "zone.home"


class StateGateway(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...


class HomeAssistantHomeLocationProvider:
    def __init__(self, gateway: StateGateway) -> None:
        self.gateway = gateway

    def get_home_coordinates(self) -> tuple[float, float] | None:
        state = self.gateway.get_state(HOME_ZONE_ENTITY_ID)
        attributes = state.get("attributes")
        if not isinstance(attributes, dict):
            return None

        latitude = _parse_coordinate(attributes.get("latitude"))
        longitude = _parse_coordinate(attributes.get("longitude"))
        if latitude is None or longitude is None:
            return None

        return latitude, longitude


def _parse_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
