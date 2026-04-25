from __future__ import annotations

from home_optimizer.domain.sensors import SENSOR_DEFINITIONS, SensorSpec


def build_sensor_specs(settings: object) -> list[SensorSpec]:
    specs: list[SensorSpec] = []
    sensor_bindings = getattr(settings, "sensors", {}) or {}

    for definition in SENSOR_DEFINITIONS:
        entity_id = sensor_bindings.get(definition.name)
        if not entity_id:
            continue

        specs.append(
            SensorSpec(
                definition=definition,
                entity_id=str(entity_id),
            )
        )

    return specs
