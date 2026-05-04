from __future__ import annotations

from typing import Literal

from pydantic import Field

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import (
    BOILER_AMBIENT_TEMPERATURE,
    BOOSTER_HEATER_ACTIVE,
    COMPRESSOR_FREQUENCY,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    DISCHARGE_TEMPERATURE,
    HP_ELECTRIC_POWER,
    HP_ELECTRIC_TOTAL_KWH,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TARGET_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_EXPORT_TOTAL_KWH,
    P1_IMPORT_TOTAL_KWH,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    PV_TOTAL_KWH,
    REFRIGERANT_CONDENSATION_TEMPERATURE,
    REFRIGERANT_LIQUID_LINE_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
)

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
        ROOM_TEMPERATURE,
        "building",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        OUTDOOR_TEMPERATURE,
        "building",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        THERMOSTAT_SETPOINT,
        "building",
        "°C",
        "ffill",
    ),
    _sensor_definition(
        SHUTTER_LIVING_ROOM,
        "building",
        "percent",
        "ffill",
    ),
    _sensor_definition(
        HP_SUPPLY_TEMPERATURE,
        "heatpump",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        HP_SUPPLY_TARGET_TEMPERATURE,
        "heatpump",
        "°C",
        "ffill",
    ),
    _sensor_definition(
        HP_RETURN_TEMPERATURE,
        "heatpump",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        HP_FLOW,
        "heatpump",
        "Lmin",
        "time_weighted_mean",
    ),
    _sensor_definition(HP_MODE, "heatpump", None, "ffill"),
    _sensor_definition(
        COMPRESSOR_FREQUENCY,
        "heatpump",
        "Hz",
        "time_weighted_mean",
    ),
    _sensor_definition(
        DEFROST_ACTIVE,
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        BOOSTER_HEATER_ACTIVE,
        "heatpump",
        "bool",
        "ffill",
    ),
    _sensor_definition(
        REFRIGERANT_CONDENSATION_TEMPERATURE,
        "heatpump",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        REFRIGERANT_LIQUID_LINE_TEMPERATURE,
        "heatpump",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        DISCHARGE_TEMPERATURE,
        "heatpump",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        DHW_TOP_TEMPERATURE,
        "dhw",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        DHW_BOTTOM_TEMPERATURE,
        "dhw",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        BOILER_AMBIENT_TEMPERATURE,
        "dhw",
        "°C",
        "interpolate",
    ),
    _sensor_definition(
        HP_ELECTRIC_POWER,
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        P1_NET_POWER,
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(
        PV_OUTPUT_POWER,
        "energy",
        "kW",
        "time_weighted_mean",
        0.001,
    ),
    _sensor_definition(PV_TOTAL_KWH, "energy", "kWh", "ffill"),
    _sensor_definition(
        HP_ELECTRIC_TOTAL_KWH,
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        P1_IMPORT_TOTAL_KWH,
        "energy",
        "kWh",
        "ffill",
    ),
    _sensor_definition(
        P1_EXPORT_TOTAL_KWH,
        "energy",
        "kWh",
        "ffill",
    ),
]
