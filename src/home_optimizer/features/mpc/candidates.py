from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import NumericPoint, NumericSeries, normalize_utc_timestamp
from home_optimizer.domain.control import ThermostatSetpointControl
from home_optimizer.domain.names import THERMOSTAT_SETPOINT


class ThermostatSetpointCandidateGenerator:
    def generate_constant_candidates(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        allowed_setpoints: list[float],
    ) -> list[ThermostatSetpointControl]:
        return [
            ThermostatSetpointControl.from_schedule(
                self._build_schedule(
                    start_time=start_time,
                    end_time=end_time,
                    interval_minutes=interval_minutes,
                    value_at=lambda _: setpoint,
                )
            )
            for setpoint in allowed_setpoints
        ]

    def generate_single_switch_candidates(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        allowed_setpoints: list[float],
        switch_times: list[datetime],
    ) -> list[ThermostatSetpointControl]:
        candidates: list[ThermostatSetpointControl] = []
        for initial_setpoint in allowed_setpoints:
            for switched_setpoint in allowed_setpoints:
                if switched_setpoint == initial_setpoint:
                    continue
                for switch_time in switch_times:
                    if switch_time <= start_time or switch_time > end_time:
                        continue
                    candidates.append(
                        ThermostatSetpointControl.from_schedule(
                            self._build_schedule(
                                start_time=start_time,
                                end_time=end_time,
                                interval_minutes=interval_minutes,
                                value_at=lambda timestamp, before=initial_setpoint, after=switched_setpoint, at=switch_time: (
                                    before if timestamp < at else after
                                ),
                            )
                        )
                    )
        return candidates

    def _build_schedule(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        value_at,
    ) -> NumericSeries:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if end_time < start_time:
            raise ValueError("end_time must be on or after start_time")

        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        timestamp = start_time
        while timestamp <= end_time:
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(timestamp),
                    value=float(value_at(timestamp)),
                )
            )
            timestamp += interval

        return NumericSeries(
            name=THERMOSTAT_SETPOINT,
            unit="degC",
            points=points,
        )
