from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ResampleMethod = Literal[
    "interpolate",
    "ffill",
    "mean",
]


@dataclass(frozen=True)
class SensorDefinition:
    config_key: str
    name: str
    category: str
    unit: str | None
    method: ResampleMethod
    conversion_factor: float = 1.0
    poll_interval_seconds: int = 5


@dataclass(frozen=True)
class SensorSpec:
    name: str
    entity_id: str
    category: str
    unit: str | None
    method: ResampleMethod


SENSOR_DEFINITIONS = [
    SensorDefinition(
        config_key="sensor_room_temperature",
        name="room_temperature",
        category="building",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_outdoor_temperature",
        name="outdoor_temperature",
        category="building",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_thermostat_setpoint",
        name="thermostat_setpoint",
        category="building",
        unit="degC",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_shutter_living_room",
        name="shutter_living_room",
        category="building",
        unit="percent",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_hp_supply_temperature",
        name="hp_supply_temperature",
        category="heatpump",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_hp_supply_target_temperature",
        name="hp_supply_target_temperature",
        category="heatpump",
        unit="degC",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_hp_return_temperature",
        name="hp_return_temperature",
        category="heatpump",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_hp_flow",
        name="hp_flow",
        category="heatpump",
        unit="L/min",
        method="mean",
    ),
    SensorDefinition(
        config_key="sensor_hp_mode",
        name="hp_mode",
        category="heatpump",
        unit=None,
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_compressor_frequency",
        name="compressor_frequency",
        category="heatpump",
        unit="Hz",
        method="mean",
    ),
    SensorDefinition(
        config_key="sensor_defrost_active",
        name="defrost_active",
        category="heatpump",
        unit="bool",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_booster_heater_active",
        name="booster_heater_active",
        category="heatpump",
        unit="bool",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_refrigerant_condensation_temperature",
        name="refrigerant_condensation_temperature",
        category="heatpump",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_refrigerant_liquid_line_temperature",
        name="refrigerant_liquid_line_temperature",
        category="heatpump",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_discharge_temperature",
        name="discharge_temperature",
        category="heatpump",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_dhw_top_temperature",
        name="dhw_top_temperature",
        category="dhw",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_dhw_bottom_temperature",
        name="dhw_bottom_temperature",
        category="dhw",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_boiler_ambient_temperature",
        name="boiler_ambient_temperature",
        category="dhw",
        unit="degC",
        method="interpolate",
    ),
    SensorDefinition(
        config_key="sensor_hp_electric_power",
        name="hp_electric_power",
        category="energy",
        unit="kW",
        method="mean",
        conversion_factor=0.001,  # W → kW
    ),
    SensorDefinition(
        config_key="sensor_p1_net_power",
        name="p1_net_power",
        category="energy",
        unit="kW",
        method="mean",
        conversion_factor=0.001,  # W → kW
    ),
    SensorDefinition(
        config_key="sensor_pv_output_power",
        name="pv_output_power",
        category="energy",
        unit="kW",
        method="mean",
        conversion_factor=0.001,  # W → kW
    ),
    SensorDefinition(
        config_key="sensor_pv_total_kwh",
        name="pv_total_kwh",
        category="energy",
        unit="kWh",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_hp_electric_total_kwh",
        name="hp_electric_total_kwh",
        category="energy",
        unit="kWh",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_p1_import_total_kwh",
        name="p1_import_total_kwh",
        category="energy",
        unit="kWh",
        method="ffill",
    ),
    SensorDefinition(
        config_key="sensor_p1_export_total_kwh",
        name="p1_export_total_kwh",
        category="energy",
        unit="kWh",
        method="ffill",
    ),
]

def build_sensor_specs(
    options,
) -> list[SensorSpec]:
    specs: list[SensorSpec] = []

    for definition in SENSOR_DEFINITIONS:
        entity_id = options.get(
            definition.config_key
        )

        if not entity_id:
            continue

        specs.append(
            SensorSpec(
                name=definition.name,
                entity_id=entity_id,
                category=definition.category,
                unit=definition.unit,
                method=definition.method,
            )
        )

    return specs