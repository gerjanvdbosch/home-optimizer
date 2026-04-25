from __future__ import annotations

from home_optimizer.domain.sensors import SENSOR_DEFINITIONS, SensorSpec


def build_sensor_specs(settings: object) -> list[SensorSpec]:
    specs: list[SensorSpec] = []

    for definition in SENSOR_DEFINITIONS:
        entity_id = getattr(settings, definition.config_key, None)
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
