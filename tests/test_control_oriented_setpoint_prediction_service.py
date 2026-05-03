from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.domain import IdentifiedModel, NumericPoint, NumericSeries
from home_optimizer.features.mpc.control_oriented import (
    StateSpaceSetpointPredictionRequest,
    StateSpaceSetpointPredictionService,
    StateSpaceThermalModel,
    StateSpaceThermalState,
)


def build_room_temperature_model() -> IdentifiedModel:
    return IdentifiedModel(
        model_kind="room_temperature",
        model_name="linear_2state_room_temperature",
        trained_at_utc=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        training_start_time_utc=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        training_end_time_utc=datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        sample_count=100,
        train_sample_count=80,
        test_sample_count=20,
        coefficients={
            "previous_room_temperature": 0.9,
            "outdoor_temperature": 0.02,
            "gti_living_room_windows_adjusted": 0.001,
            "floor_heat_state": 0.5,
        },
        intercept=0.5,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )


def build_thermal_output_model() -> IdentifiedModel:
    return IdentifiedModel(
        model_kind="thermal_output",
        model_name="linear_1step_thermal_output",
        trained_at_utc=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        training_start_time_utc=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        training_end_time_utc=datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        sample_count=100,
        train_sample_count=80,
        test_sample_count=20,
        coefficients={
            "previous_thermal_output": 0.1,
            "previous_heating_demand": 1.5,
            "previous_floor_heat_state": 0.0,
            "outdoor_temperature": 0.0,
            "hp_supply_target_temperature": 0.0,
            "_active_intercept": 1.0,
            "_active_target_threshold": 0.05,
            "active::previous_thermal_output": 0.0,
            "active::previous_heating_demand": 1.0,
            "active::previous_floor_heat_state": 0.0,
            "active::outdoor_temperature": 0.0,
            "active::hp_supply_target_temperature": 0.0,
        },
        intercept=0.0,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="thermal_output",
    )


def series(name: str, unit: str | None, points: list[tuple[str, float]]) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=unit,
        points=[NumericPoint(timestamp=timestamp, value=value) for timestamp, value in points],
    )


def test_state_space_setpoint_prediction_service_simulates_actuator_and_plant() -> None:
    service = StateSpaceSetpointPredictionService()
    room_model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())
    thermal_output_model = build_thermal_output_model()

    result = service.predict(
        thermal_model=room_model,
        thermal_output_model=thermal_output_model,
        request=StateSpaceSetpointPredictionRequest(
            start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
            initial_state=StateSpaceThermalState(room_temperature=19.0, floor_heat_state=0.0),
            initial_thermal_output=0.0,
            thermostat_setpoint_schedule=series(
                "thermostat_setpoint",
                "degC",
                [
                    ("2026-05-03T10:15:00+00:00", 21.0),
                    ("2026-05-03T10:30:00+00:00", 21.0),
                ],
            ),
            outdoor_temperature_series=series(
                "outdoor_temperature",
                "degC",
                [
                    ("2026-05-03T10:15:00+00:00", 10.0),
                    ("2026-05-03T10:30:00+00:00", 10.0),
                ],
            ),
            solar_gain_series=series(
                "gti_living_room_windows_adjusted",
                "Wm2",
                [
                    ("2026-05-03T10:15:00+00:00", 0.0),
                    ("2026-05-03T10:30:00+00:00", 0.0),
                ],
            ),
            supply_target_temperature_series=series(
                "hp_supply_target_temperature",
                "degC",
                [
                    ("2026-05-03T10:15:00+00:00", 24.0),
                    ("2026-05-03T10:30:00+00:00", 24.0),
                ],
            ),
        ),
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.thermal_output_model_name == "linear_1step_thermal_output"
    assert [point.value for point in result.thermal_output.points] == [3.0, 5.032499999999996]
    assert [point.value for point in result.floor_heat_state.points] == pytest.approx(
        [0.09, 0.2382749999999999]
    )
    assert [point.value for point in result.room_temperature.points] == pytest.approx(
        [17.845, 16.8796375]
    )
