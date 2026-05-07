from __future__ import annotations

from typing import Literal

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import (
    BOOSTER_HEATER_ACTIVE,
    COMPRESSOR_FREQUENCY,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    HP_ELECTRIC_POWER,
    HP_MODE,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
)

SignalRole = Literal[
    "output",
    "control_input",
    "measured_disturbance",
    "forecast_disturbance",
    "measured_response",
    "state_candidate",
    "context",
]


class ModelingSignalDefinition(DomainModel):
    name: str
    column_name: str
    role: SignalRole
    required: bool = True
    forecast_required: bool = False
    controllable: bool = False
    is_text: bool = False


MODELING_SIGNAL_DEFINITIONS = [
    ModelingSignalDefinition(
        name=ROOM_TEMPERATURE,
        column_name="room_temperature_c",
        role="output",
    ),
    ModelingSignalDefinition(
        name=DHW_TOP_TEMPERATURE,
        column_name="dhw_top_temperature_c",
        role="output",
    ),
    ModelingSignalDefinition(
        name=DHW_BOTTOM_TEMPERATURE,
        column_name="dhw_bottom_temperature_c",
        role="state_candidate",
        required=False,
    ),
    ModelingSignalDefinition(
        name=OUTDOOR_TEMPERATURE,
        column_name="outdoor_temperature_c",
        role="forecast_disturbance",
        forecast_required=True,
    ),
    ModelingSignalDefinition(
        name=THERMOSTAT_SETPOINT,
        column_name="thermostat_setpoint_c",
        role="control_input",
        forecast_required=True,
        controllable=True,
    ),
    ModelingSignalDefinition(
        name=SHUTTER_LIVING_ROOM,
        column_name="shutter_living_room_pct",
        role="measured_disturbance",
        required=False,
    ),
    ModelingSignalDefinition(
        name=HP_ELECTRIC_POWER,
        column_name="hp_electric_power_kw",
        role="measured_response",
    ),
    ModelingSignalDefinition(
        name=COMPRESSOR_FREQUENCY,
        column_name="compressor_frequency_hz",
        role="measured_response",
        required=False,
    ),
    ModelingSignalDefinition(
        name=DEFROST_ACTIVE,
        column_name="defrost_active",
        role="measured_response",
        required=False,
    ),
    ModelingSignalDefinition(
        name=BOOSTER_HEATER_ACTIVE,
        column_name="booster_heater_active",
        role="measured_response",
        required=False,
    ),
    ModelingSignalDefinition(
        name=PV_OUTPUT_POWER,
        column_name="pv_output_power_kw",
        role="measured_disturbance",
        required=False,
    ),
    ModelingSignalDefinition(
        name=P1_NET_POWER,
        column_name="p1_net_power_kw",
        role="measured_response",
        required=False,
    ),
    ModelingSignalDefinition(
        name=HP_MODE,
        column_name="hp_mode",
        role="measured_response",
        required=False,
        is_text=True,
    ),
]


def modeling_signal_definitions() -> list[ModelingSignalDefinition]:
    return list(MODELING_SIGNAL_DEFINITIONS)


def modeling_numeric_signal_definitions() -> list[ModelingSignalDefinition]:
    return [
        signal
        for signal in MODELING_SIGNAL_DEFINITIONS
        if not signal.is_text
    ]


def modeling_text_signal_definitions() -> list[ModelingSignalDefinition]:
    return [
        signal
        for signal in MODELING_SIGNAL_DEFINITIONS
        if signal.is_text
    ]


def modeling_names() -> list[str]:
    return [
        signal.name
        for signal in MODELING_SIGNAL_DEFINITIONS
    ]


def sensors_by_role(role: SignalRole) -> list[ModelingSignalDefinition]:
    return [
        signal
        for signal in MODELING_SIGNAL_DEFINITIONS
        if signal.role == role
    ]