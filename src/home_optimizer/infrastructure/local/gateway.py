from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.types import JsonDict


class LocalJsonGateway:
    def __init__(
        self,
        state_path: str,
        specs: list[SensorSpec],
    ) -> None:
        self.state_path = Path(state_path)
        self.entity_to_name = {spec.entity_id: spec.name for spec in specs}
        self.name_to_entity = {spec.name: spec.entity_id for spec in specs}

    def close(self) -> None:
        return None

    def get_state(self, entity_id: str) -> dict[str, Any]:
        sensor_name = self.entity_to_name.get(entity_id)
        state = self._load_sensor_state()
        value = state.get(sensor_name) if sensor_name else None
        timestamp = utc_now().isoformat()

        return {
            "entity_id": entity_id,
            "state": "unavailable" if value is None else value,
            "last_changed": timestamp,
            "last_updated": timestamp,
        }

    def get_states(self) -> list[dict[str, Any]]:
        return [self.get_state(entity_id) for entity_id in self.name_to_entity.values()]

    def get_location(self) -> tuple[float, float] | None:
        data = self._load_state_file()
        location = data.get("location")
        if not isinstance(location, dict):
            return None

        latitude = _parse_coordinate(location.get("latitude"))
        longitude = _parse_coordinate(location.get("longitude"))
        if latitude is None or longitude is None:
            return None

        return latitude, longitude

    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        minimal_response: bool = True,
    ) -> list[dict[str, Any]]:
        return []

    def _load_sensor_state(self) -> JsonDict:
        data = self._load_state_file()
        sensors = data.get("sensors", data)
        if not isinstance(sensors, dict):
            raise ValueError(f"Invalid sensors section in: {self.state_path}")

        return sensors

    def _load_state_file(self) -> JsonDict:
        if not self.state_path.exists():
            return {}

        with self.state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid local state file: {self.state_path}")

        return data


def _parse_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
