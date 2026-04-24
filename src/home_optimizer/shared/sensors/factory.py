from __future__ import annotations

from typing import Any

from home_optimizer.shared.sensors.definitions import SENSOR_DEFINITIONS, SensorSpec


def build_sensor_specs(options: dict[str, Any]) -> list[SensorSpec]:
    specs: list[SensorSpec] = []

    for definition in SENSOR_DEFINITIONS:
        entity_id = options.get(definition.config_key)
        if not entity_id:
            continue

        specs.append(
            SensorSpec(
                name=definition.name,
                entity_id=str(entity_id),
                category=definition.category,
                unit=definition.unit,
                method=definition.method,
                conversion_factor=definition.conversion_factor,
            )
        )

    return specs
