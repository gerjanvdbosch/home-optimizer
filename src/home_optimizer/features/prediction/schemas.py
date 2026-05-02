from __future__ import annotations

from home_optimizer.domain import (
    DomainModel,
    NumericSeries,
    ShutterPositionControl,
    ThermostatSetpointControl,
)


class RoomTemperatureControlInputs(DomainModel):
    thermostat_setpoint: ThermostatSetpointControl
    shutter_position: ShutterPositionControl | None = None


class RoomTemperaturePrediction(DomainModel):
    model_name: str
    interval_minutes: int
    target_name: str
    room_temperature: NumericSeries


class RoomTemperaturePredictionComparison(DomainModel):
    model_name: str
    interval_minutes: int
    target_name: str
    predicted_room_temperature: NumericSeries
    actual_room_temperature: NumericSeries
