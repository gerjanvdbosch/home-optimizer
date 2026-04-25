from __future__ import annotations

from typing import Any

from home_optimizer.infrastructure.home_assistant.location import HomeAssistantHomeLocationProvider


class FakeStateGateway:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.requested_entity_ids: list[str] = []

    def get_state(self, entity_id: str) -> dict[str, Any]:
        self.requested_entity_ids.append(entity_id)
        return self.state


def test_home_assistant_home_location_provider_reads_zone_home_coordinates() -> None:
    gateway = FakeStateGateway(
        {"attributes": {"latitude": "52.09", "longitude": 5.12}},
    )
    provider = HomeAssistantHomeLocationProvider(gateway)

    assert provider.get_home_coordinates() == (52.09, 5.12)
    assert gateway.requested_entity_ids == ["zone.home"]


def test_home_assistant_home_location_provider_returns_none_for_missing_coordinates() -> None:
    gateway = FakeStateGateway({"attributes": {"latitude": None, "longitude": 5.12}})
    provider = HomeAssistantHomeLocationProvider(gateway)

    assert provider.get_home_coordinates() is None
