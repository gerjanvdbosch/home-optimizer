from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Protocol

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    ShutterPositionControl,
    ThermostatSetpointControl,
    latest_value_at,
    normalize_utc_timestamp,
)

from .schemas import (
    DEFAULT_MPC_HORIZON_HOURS,
    ThermostatSetpointMpcClosedLoopDayResult,
    ThermostatSetpointMpcClosedLoopResult,
    ThermostatSetpointMpcClosedLoopStepResult,
    ThermostatSetpointMpcPlanRequest,
)


class ScheduleReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...


class MpcPlannerRunner(Protocol):
    def propose_plan(
        self,
        request: ThermostatSetpointMpcPlanRequest,
        *,
        shutter_position: ShutterPositionControl | None = None,
    ): ...


class ThermostatSetpointMpcClosedLoopService:
    def __init__(
        self,
        reader: ScheduleReader,
        planner: MpcPlannerRunner,
    ) -> None:
        self.reader = reader
        self.planner = planner

    def evaluate_by_day(
        self,
        *,
        start_date: date,
        end_date: date,
        allowed_setpoints: list[float],
        horizon_hours: int = DEFAULT_MPC_HORIZON_HOURS,
        interval_minutes: int = 15,
        switch_interval_hours: int = 2,
        timezone_info: tzinfo | None = None,
        comfort_min_temperature: float = 19.0,
        comfort_max_temperature: float = 21.0,
        setpoint_change_penalty: float = 0.1,
    ) -> ThermostatSetpointMpcClosedLoopResult:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be greater than zero")
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if switch_interval_hours <= 0:
            raise ValueError("switch_interval_hours must be greater than zero")
        if not allowed_setpoints:
            raise ValueError("allowed_setpoints must not be empty")
        if comfort_min_temperature > comfort_max_temperature:
            raise ValueError("comfort_min_temperature must be <= comfort_max_temperature")

        local_timezone = timezone_info or datetime.now().astimezone().tzinfo or timezone.utc
        interval = timedelta(minutes=interval_minutes)
        current_date = start_date
        day_results: list[ThermostatSetpointMpcClosedLoopDayResult] = []
        model_name = ""

        while current_date <= end_date:
            day_start = datetime.combine(current_date, time.min, tzinfo=local_timezone)
            full_day_end = day_start + timedelta(days=1) - interval
            try:
                schedules = self.reader.read_series(
                    names=[THERMOSTAT_SETPOINT, SHUTTER_LIVING_ROOM],
                    start_time=day_start - interval,
                    end_time=full_day_end,
                )
                schedules_by_name = {series.name: series for series in schedules}
                measured_setpoint_schedule = schedules_by_name.get(
                    THERMOSTAT_SETPOINT,
                    NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=[]),
                )
                shutter_schedule = schedules_by_name.get(
                    SHUTTER_LIVING_ROOM,
                    NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[]),
                )
                if not measured_setpoint_schedule.points:
                    raise ValueError("no measured thermostat setpoint available")
                if not shutter_schedule.points:
                    raise ValueError("no measured shutter series available")

                shutter_position = ShutterPositionControl.from_schedule(shutter_schedule)
                step_results: list[ThermostatSetpointMpcClosedLoopStepResult] = []
                applied_points: list[NumericPoint] = []
                predicted_points: list[NumericPoint] = []
                comfort_costs: list[float] = []
                change_costs: list[float] = []
                total_costs: list[float] = []

                step_start = day_start
                while step_start < full_day_end:
                    horizon_end = min(
                        step_start + timedelta(hours=horizon_hours) - interval,
                        full_day_end,
                    )
                    plan = self.planner.propose_plan(
                        ThermostatSetpointMpcPlanRequest(
                            start_time=step_start,
                            end_time=horizon_end,
                            interval_minutes=interval_minutes,
                            allowed_setpoints=allowed_setpoints,
                            switch_times=self._build_switch_times(
                                start_time=step_start,
                                end_time=horizon_end,
                                switch_interval_hours=switch_interval_hours,
                            ),
                            comfort_min_temperature=comfort_min_temperature,
                            comfort_max_temperature=comfort_max_temperature,
                            setpoint_change_penalty=setpoint_change_penalty,
                        ),
                        shutter_position=shutter_position,
                    )
                    model_name = model_name or plan.model_name
                    first_setpoint = latest_value_at(
                        plan.best_candidate.thermostat_setpoint_schedule.points,
                        normalize_utc_timestamp(step_start),
                    )
                    if first_setpoint is None:
                        raise ValueError("best candidate is missing setpoint at current step")
                    if not plan.best_candidate.predicted_room_temperature.points:
                        raise ValueError("best candidate is missing predicted room temperature")

                    predicted_next_point = plan.best_candidate.predicted_room_temperature.points[0]
                    applied_points.append(
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(step_start),
                            value=float(first_setpoint),
                        )
                    )
                    predicted_points.append(predicted_next_point)
                    total_costs.append(plan.best_candidate.total_cost)
                    comfort_costs.append(plan.best_candidate.comfort_violation_cost)
                    change_costs.append(plan.best_candidate.setpoint_change_cost)
                    step_results.append(
                        ThermostatSetpointMpcClosedLoopStepResult(
                            step_start_time=step_start,
                            applied_setpoint=float(first_setpoint),
                            best_candidate_name=plan.best_candidate.candidate_name,
                            best_candidate_total_cost=plan.best_candidate.total_cost,
                            predicted_next_room_temperature=predicted_next_point.value,
                        )
                    )
                    step_start += interval

                predicted_values = [point.value for point in predicted_points]
                day_results.append(
                    ThermostatSetpointMpcClosedLoopDayResult(
                        day=current_date,
                        horizon_hours=horizon_hours,
                        interval_minutes=interval_minutes,
                        applied_thermostat_setpoint_schedule=NumericSeries(
                            name=THERMOSTAT_SETPOINT,
                            unit="degC",
                            points=applied_points,
                        ),
                        measured_thermostat_setpoint_schedule=measured_setpoint_schedule,
                        predicted_room_temperature=NumericSeries(
                            name=ROOM_TEMPERATURE,
                            unit="degC",
                            points=predicted_points,
                        ),
                        average_total_cost=sum(total_costs) / len(total_costs),
                        average_comfort_violation_cost=sum(comfort_costs) / len(comfort_costs),
                        average_setpoint_change_cost=sum(change_costs) / len(change_costs),
                        minimum_predicted_temperature=min(predicted_values) if predicted_values else None,
                        maximum_predicted_temperature=max(predicted_values) if predicted_values else None,
                        under_comfort_count=sum(
                            1 for value in predicted_values if value < comfort_min_temperature
                        ),
                        over_comfort_count=sum(
                            1 for value in predicted_values if value > comfort_max_temperature
                        ),
                        step_results=step_results,
                    )
                )
            except ValueError as error:
                day_results.append(
                    ThermostatSetpointMpcClosedLoopDayResult(
                        day=current_date,
                        horizon_hours=horizon_hours,
                        interval_minutes=interval_minutes,
                        applied_thermostat_setpoint_schedule=NumericSeries(
                            name=THERMOSTAT_SETPOINT,
                            unit="degC",
                            points=[],
                        ),
                        measured_thermostat_setpoint_schedule=NumericSeries(
                            name=THERMOSTAT_SETPOINT,
                            unit="degC",
                            points=[],
                        ),
                        predicted_room_temperature=NumericSeries(
                            name=ROOM_TEMPERATURE,
                            unit="degC",
                            points=[],
                        ),
                        average_total_cost=0.0,
                        average_comfort_violation_cost=0.0,
                        average_setpoint_change_cost=0.0,
                        minimum_predicted_temperature=None,
                        maximum_predicted_temperature=None,
                        under_comfort_count=0,
                        over_comfort_count=0,
                        step_results=[],
                        error=str(error),
                    )
                )

            current_date += timedelta(days=1)

        successful_results = [result for result in day_results if result.error is None]
        return ThermostatSetpointMpcClosedLoopResult(
            model_name=model_name,
            interval_minutes=interval_minutes,
            horizon_hours=horizon_hours,
            start_date=start_date,
            end_date=end_date,
            total_days=len(day_results),
            successful_days=len(successful_results),
            failed_days=len(day_results) - len(successful_results),
            average_total_cost=(
                sum(result.average_total_cost for result in successful_results) / len(successful_results)
                if successful_results
                else None
            ),
            average_comfort_violation_cost=(
                sum(result.average_comfort_violation_cost for result in successful_results)
                / len(successful_results)
                if successful_results
                else None
            ),
            average_setpoint_change_cost=(
                sum(result.average_setpoint_change_cost for result in successful_results)
                / len(successful_results)
                if successful_results
                else None
            ),
            day_results=day_results,
        )

    @staticmethod
    def _build_switch_times(
        *,
        start_time: datetime,
        end_time: datetime,
        switch_interval_hours: int,
    ) -> list[datetime]:
        switch_times: list[datetime] = []
        cursor = start_time + timedelta(hours=switch_interval_hours)
        while cursor <= end_time:
            switch_times.append(cursor)
            cursor += timedelta(hours=switch_interval_hours)
        return switch_times
