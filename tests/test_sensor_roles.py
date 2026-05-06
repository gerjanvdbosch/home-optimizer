from __future__ import annotations

from home_optimizer.domain import (
    COMPRESSOR_FREQUENCY,
    DHW_TOP_TEMPERATURE,
    HP_ELECTRIC_POWER,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SENSOR_DEFINITIONS,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
)


def test_sensor_definitions_expose_signal_roles_and_flags() -> None:
    definitions_by_name = {
        definition.name: definition
        for definition in SENSOR_DEFINITIONS
    }

    assert definitions_by_name[ROOM_TEMPERATURE].role == "output"
    assert definitions_by_name[DHW_TOP_TEMPERATURE].role == "output"
    assert definitions_by_name[THERMOSTAT_SETPOINT].role == "control_input"
    assert definitions_by_name[OUTDOOR_TEMPERATURE].role == "forecast_disturbance"
    assert definitions_by_name[SHUTTER_LIVING_ROOM].role == "measured_disturbance"
    assert definitions_by_name[PV_OUTPUT_POWER].role == "measured_disturbance"
    assert definitions_by_name[HP_ELECTRIC_POWER].role == "measured_response"
    assert definitions_by_name[COMPRESSOR_FREQUENCY].role == "measured_response"
    assert definitions_by_name[P1_NET_POWER].role == "measured_response"

    assert definitions_by_name[OUTDOOR_TEMPERATURE].forecast_required is True
    assert definitions_by_name[THERMOSTAT_SETPOINT].controllable is True
    assert definitions_by_name[ROOM_TEMPERATURE].forecast_required is False
    assert definitions_by_name[ROOM_TEMPERATURE].controllable is False
