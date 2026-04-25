from __future__ import annotations

from typing import Literal

from pydantic import Field

from home_optimizer.domain.models import DomainModel

ResampleMethod = Literal["interpolate", "ffill", "mean", "time_weighted_mean"]


class SensorDefinition(DomainModel):
    name: str
    category: str
    unit: str | None
    method: ResampleMethod
    conversion_factor: float = 1.0
    poll_interval_seconds: int = Field(default=5, gt=0)


class SensorSpec(DomainModel):
    definition: SensorDefinition
    entity_id: str

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def category(self) -> str:
        return self.definition.category

    @property
    def unit(self) -> str | None:
        return self.definition.unit

    @property
    def method(self) -> ResampleMethod:
        return self.definition.method

    @property
    def conversion_factor(self) -> float:
        return self.definition.conversion_factor

    @property
    def poll_interval_seconds(self) -> int:
        return self.definition.poll_interval_seconds


def _sensor_definition(
    name: str,
    category: str,
    unit: str | None,
    method: ResampleMethod,
    conversion_factor: float = 1.0,
    poll_interval_seconds: int = 5,
) -> SensorDefinition:
    return SensorDefinition(
        name=name,
        category=category,
        unit=unit,
        method=method,
        conversion_factor=conversion_factor,
        poll_interval_seconds=poll_interval_seconds,
    )


SENSOR_DEFINITIONS = [
    _sensor_definition(
        "room_temperature",
        "building",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "outdoor_temperature",
        "building",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "thermostat_setpoint",
        "building",
        "degC",
        "ffill",
    ),
    _sensor_definition(
        "shutter_living_room",
        "building",
        "percent",
        "ffill",
    ),
    _sensor_definition(
        "hp_supply_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "hp_supply_target_temperature",
        "heatpump",
        "degC",
        "ffill",
    ),
    _sensor_definition(
        "hp_return_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "hp_flow",
        "heatpump",
        "Lmin",
        "time_weighted_mean",
    ),
    _sensor_definition("hp_mode", "heatpump", None, "ffill"),
    _sensor_definition(
        "compressor_frequency",
        "heatpump",
        "Hz",
        "time_weighted_mean",
    ),
    _sensor_definition(
        "defrost_active",
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        "booster_heater_active",
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        "refrigerant_condensation_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "refrigerant_liquid_line_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "discharge_temperature",
        "heatpump",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "dhw_top_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "dhw_bottom_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "boiler_ambient_temperature",
        "dhw",
        "degC",
        "interpolate",
    ),
    _sensor_definition(
        "hp_electric_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        "p1_net_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        "pv_output_power",
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition("pv_total_kwh", "energy", "kWh", "ffill"),
    _sensor_definition(
        "hp_electric_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        "p1_import_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        "p1_export_total_kwh",
        "energy",
        "kWh",
        "ffill",
    ),
]
