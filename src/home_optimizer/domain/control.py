from __future__ import annotations

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import SHUTTER_LIVING_ROOM, THERMOSTAT_SETPOINT
from home_optimizer.domain.series import NumericSeries


class ThermostatSetpointControl(DomainModel):
    schedule: NumericSeries

    @classmethod
    def from_schedule(cls, schedule: NumericSeries) -> "ThermostatSetpointControl":
        if schedule.name != THERMOSTAT_SETPOINT:
            raise ValueError(
                f"thermostat setpoint control must use series name {THERMOSTAT_SETPOINT!r}"
            )
        return cls(schedule=schedule)


class ShutterPositionControl(DomainModel):
    schedule: NumericSeries

    @classmethod
    def from_schedule(cls, schedule: NumericSeries) -> "ShutterPositionControl":
        if schedule.name != SHUTTER_LIVING_ROOM:
            raise ValueError(
                f"shutter position control must use series name {SHUTTER_LIVING_ROOM!r}"
            )
        return cls(schedule=schedule)
