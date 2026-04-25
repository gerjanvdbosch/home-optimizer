from __future__ import annotations

import json

from home_optimizer.domain.sensors import SensorDefinition, SensorSpec
from home_optimizer.infrastructure.local.gateway import LocalJsonGateway


def test_local_json_gateway_reads_sensor_state_by_entity_id(tmp_path) -> None:
    state_path = tmp_path / "local.json"
    state_path.write_text(
        json.dumps({"sensors": {"room_temperature": 19.8}}),
        encoding="utf-8",
    )
    spec = SensorSpec(
        definition=SensorDefinition(
            name="room_temperature",
            category="building",
            unit="degC",
            method="mean",
        ),
        entity_id="sensor.local_room",
    )
    gateway = LocalJsonGateway(str(state_path), specs=[spec])

    state = gateway.get_state("sensor.local_room")

    assert state["entity_id"] == "sensor.local_room"
    assert state["state"] == 19.8
    assert state["last_updated"].endswith("+00:00")


def test_local_json_gateway_treats_missing_sensor_as_unavailable(tmp_path) -> None:
    state_path = tmp_path / "local.json"
    state_path.write_text(json.dumps({"sensors": {}}), encoding="utf-8")
    spec = SensorSpec(
        definition=SensorDefinition(
            name="room_temperature",
            category="building",
            unit="degC",
            method="mean",
        ),
        entity_id="sensor.local_room",
    )
    gateway = LocalJsonGateway(str(state_path), specs=[spec])

    assert gateway.get_state("sensor.local_room")["state"] == "unavailable"

