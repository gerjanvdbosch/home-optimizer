from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo

from home_optimizer.domain import (
    NumericSeries,
    ROOM_TEMPERATURE,
    THERMOSTAT_SETPOINT,
    SHUTTER_LIVING_ROOM,
    ShutterPositionControl,
    ThermostatSetpointControl,
)
from home_optimizer.features.identification.room_temperature.model import MODEL_KIND
from home_optimizer.features.prediction.ports import IdentifiedModelReader, PredictionDataReader
from home_optimizer.features.prediction.schemas import RoomTemperatureControlInputs
from home_optimizer.features.prediction.service import RoomTemperaturePredictionService

from .metrics import prediction_error_summary
from .schemas import RoomTemperatureBacktestDayResult, RoomTemperatureBacktestResult


class RoomTemperatureBacktestingService:
    def __init__(
        self,
        reader: PredictionDataReader,
        model_repository: IdentifiedModelReader,
        prediction_service: RoomTemperaturePredictionService,
    ) -> None:
        self.reader = reader
        self.model_repository = model_repository
        self.prediction_service = prediction_service

    def backtest_by_day(
        self,
        *,
        start_date: date,
        end_date: date,
        horizon_hours: int = 24,
        timezone_info: tzinfo | None = None,
        comfort_min_temperature: float | None = None,
        comfort_max_temperature: float | None = None,
    ) -> RoomTemperatureBacktestResult:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be greater than zero")
        if (
            comfort_min_temperature is not None
            and comfort_max_temperature is not None
            and comfort_min_temperature > comfort_max_temperature
        ):
            raise ValueError("comfort_min_temperature must be <= comfort_max_temperature")

        model = self.model_repository.latest(model_kind=MODEL_KIND)
        if model is None:
            raise ValueError("no stored room temperature model available")

        interval = timedelta(minutes=model.interval_minutes)
        local_timezone = timezone_info or datetime.now().astimezone().tzinfo or timezone.utc
        current_date = start_date
        day_results: list[RoomTemperatureBacktestDayResult] = []

        while current_date <= end_date:
            day_start = datetime.combine(current_date, time.min, tzinfo=local_timezone)
            full_day_end = day_start + timedelta(days=1) - interval
            day_end = min(day_start + timedelta(hours=horizon_hours) - interval, full_day_end)
            schedule_start = day_start - interval

            try:
                measured_schedules = self.reader.read_series(
                    names=[THERMOSTAT_SETPOINT, SHUTTER_LIVING_ROOM],
                    start_time=schedule_start,
                    end_time=day_end,
                )
                schedules_by_name = {series.name: series for series in measured_schedules}
                thermostat_schedule = schedules_by_name.get(
                    THERMOSTAT_SETPOINT,
                    NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=[]),
                )
                shutter_schedule = schedules_by_name.get(
                    SHUTTER_LIVING_ROOM,
                    NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[]),
                )
                if not thermostat_schedule.points:
                    raise ValueError("no measured thermostat setpoint available")
                if not shutter_schedule.points:
                    raise ValueError("no measured shutter series available")

                comparison = self.prediction_service.predict_vs_actual(
                    start_time=day_start,
                    end_time=day_end,
                    control_inputs=RoomTemperatureControlInputs(
                        thermostat_setpoint=ThermostatSetpointControl.from_schedule(
                            thermostat_schedule
                        ),
                        shutter_position=ShutterPositionControl.from_schedule(shutter_schedule),
                    ),
                )
                overlap_count, rmse, bias, max_absolute_error = prediction_error_summary(
                    predicted=comparison.predicted_room_temperature,
                    actual=comparison.actual_room_temperature,
                )
                predicted_values = [
                    point.value for point in comparison.predicted_room_temperature.points
                ]
                minimum_predicted_temperature = (
                    min(predicted_values) if predicted_values else None
                )
                maximum_predicted_temperature = (
                    max(predicted_values) if predicted_values else None
                )
                under_comfort_count = None
                over_comfort_count = None
                if (
                    comfort_min_temperature is not None
                    and comfort_max_temperature is not None
                ):
                    under_comfort_count = sum(
                        1 for value in predicted_values if value < comfort_min_temperature
                    )
                    over_comfort_count = sum(
                        1 for value in predicted_values if value > comfort_max_temperature
                    )

                day_results.append(
                    RoomTemperatureBacktestDayResult(
                        day=current_date,
                        horizon_hours=horizon_hours,
                        overlap_count=overlap_count,
                        rmse=rmse,
                        bias=bias,
                        max_absolute_error=max_absolute_error,
                        minimum_predicted_temperature=minimum_predicted_temperature,
                        maximum_predicted_temperature=maximum_predicted_temperature,
                        under_comfort_count=under_comfort_count,
                        over_comfort_count=over_comfort_count,
                    )
                )
            except ValueError as error:
                day_results.append(
                    RoomTemperatureBacktestDayResult(
                        day=current_date,
                        horizon_hours=horizon_hours,
                        overlap_count=0,
                        rmse=None,
                        bias=None,
                        max_absolute_error=None,
                        minimum_predicted_temperature=None,
                        maximum_predicted_temperature=None,
                        error=str(error),
                    )
                )

            current_date += timedelta(days=1)

        successful_results = [result for result in day_results if result.error is None]
        rmse_values = [result.rmse for result in successful_results if result.rmse is not None]
        bias_values = [result.bias for result in successful_results if result.bias is not None]
        max_error_values = [
            result.max_absolute_error
            for result in successful_results
            if result.max_absolute_error is not None
        ]
        worst_day = None
        if rmse_values:
            worst_result = max(
                (
                    result
                    for result in successful_results
                    if result.rmse is not None
                ),
                key=lambda result: float(result.rmse),
            )
            worst_day = worst_result.day

        return RoomTemperatureBacktestResult(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            horizon_hours=horizon_hours,
            start_date=start_date,
            end_date=end_date,
            total_days=len(day_results),
            successful_days=len(successful_results),
            failed_days=len(day_results) - len(successful_results),
            average_rmse=(sum(rmse_values) / len(rmse_values)) if rmse_values else None,
            average_bias=(sum(bias_values) / len(bias_values)) if bias_values else None,
            average_max_absolute_error=(
                sum(max_error_values) / len(max_error_values)
            )
            if max_error_values
            else None,
            worst_day_by_rmse=worst_day,
            day_results=day_results,
        )
