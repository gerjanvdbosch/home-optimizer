from __future__ import annotations

from typing import Literal

from pydantic import Field

from home_optimizer.domain.models import DomainModel

ResampleMethod = Literal["interpolate", "ffill", "mean", "time_weighted_mean"]


class SensorDefinition(DomainModel):
    config_key: str
    name: str
    category: str
    unit: str | None
    method: ResampleMethod
    conversion_factor: float = 1.0
    poll_interval_seconds: int = Field(default=5, gt=0)


class SensorSpec(DomainModel):
    name: str
    entity_id: str
    category: str
    unit: str | None
    method: ResampleMethod
    conversion_factor: float = 1.0


def _sensor_definition(
    config_key: str,
    name: str,
    category: str,
    unit: str | None,
    method: ResampleMethod,
    conversion_factor: float = 1.0,
    poll_interval_seconds: int = 5,
) -> SensorDefinition:
    return SensorDefinition(
        config_key=config_key,
        name=name,
        category=category,
        unit=unit,
        method=method,
        conversion_factor=conversion_factor,
        poll_interval_seconds=poll_interval_seconds,
    )


SENSOR_DEFINITIONS = [
    _sensor_definition(
        "sensor_room_temperature",
        "room_temperature",
        "building",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_outdoor_temperature",
        "outdoor_temperature",
        "building",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_thermostat_setpoint",
        "thermostat_setpoint",
        "building",
        "degC",
        "ffill",
    ),
    _sensor_definition(
        "sensor_shutter_living_room",
        "shutter_living_room",
        "building",
        "percent",
        "ffill",
    ),
    _sensor_definition(
        "sensor_hp_supply_temperature",
        "hp_supply_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_hp_supply_target_temperature",
        "hp_supply_target_temperature",
        "heatpump",
        "degC",
        "ffill",
    ),
    _sensor_definition(
        "sensor_hp_return_temperature",
        "hp_return_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_hp_flow",
        "hp_flow",
        "heatpump",
        "Lmin",
        "time_weighted_mean",
    ),
    _sensor_definition("sensor_hp_mode", "hp_mode", "heatpump", None, "ffill"),
    _sensor_definition(
        "sensor_compressor_frequency",
        "compressor_frequency",
        "heatpump",
        "Hz",
        "time_weighted_mean",
    ),
    _sensor_definition(
        "sensor_defrost_active",
        "defrost_active",
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        "sensor_booster_heater_active",
        "booster_heater_active",
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        "sensor_refrigerant_condensation_temperature",
        "refrigerant_condensation_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_refrigerant_liquid_line_temperature",
        "refrigerant_liquid_line_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_discharge_temperature",
        "discharge_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_dhw_top_temperature",
        "dhw_top_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_dhw_bottom_temperature",
        "dhw_bottom_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_boiler_ambient_temperature",
        "boiler_ambient_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "sensor_hp_electric_power",
        "hp_electric_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        "sensor_p1_net_power",
        "p1_net_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        "sensor_pv_output_power",
        "pv_output_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition("sensor_pv_total_kwh", "pv_total_kwh", "energy", "kWh", "ffill"),
    _sensor_definition(
        "sensor_hp_electric_total_kwh",
        "hp_electric_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        "sensor_p1_import_total_kwh",
        "p1_import_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        "sensor_p1_export_total_kwh",
        "p1_export_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
]
