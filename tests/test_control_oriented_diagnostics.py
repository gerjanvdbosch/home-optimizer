from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.domain import IdentifiedModel, NumericPoint, NumericSeries
from home_optimizer.features.mpc.control_oriented import StateSpaceActuatorSensitivityService
from home_optimizer.features.prediction.service import _PreparedPredictionContext


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


class FakePredictionService:
    def prepare_prediction_context(self, *, start_time, end_time, shutter_position=None, model_name=None):
        return _PreparedPredictionContext(
            model=build_room_temperature_model(),
            interval=(end_time - start_time) / 2,
            start_time=start_time,
            end_time=end_time,
            outdoor_forecast=series(
                "outdoor_temperature",
                "degC",
                [
                    ("2026-05-03T10:15:00+00:00", 10.0),
                    ("2026-05-03T10:30:00+00:00", 10.0),
                ],
            ),
            adjusted_gti=series(
                "gti_living_room_windows_adjusted",
                "Wm2",
                [
                    ("2026-05-03T10:15:00+00:00", 0.0),
                    ("2026-05-03T10:30:00+00:00", 0.0),
                ],
            ),
            initial_room_temperature=19.0,
            initial_floor_heat_state=0.0,
            thermal_output_model=build_thermal_output_model(),
            initial_thermal_output=0.0,
            supply_target_temperature_series=series(
                "hp_supply_target_temperature",
                "degC",
                [
                    ("2026-05-03T10:15:00+00:00", 24.0),
                    ("2026-05-03T10:30:00+00:00", 24.0),
                ],
            ),
        )


def test_actuator_sensitivity_rows_reflect_setpoint_influence() -> None:
    service = StateSpaceActuatorSensitivityService(
        prediction_service=FakePredictionService(),
    )

    result = service.inspect(
        start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
        setpoints=[19.0, 21.0],
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.thermal_output_model_name == "linear_1step_thermal_output"
    assert len(result.rows) == 2
    assert result.rows[0].thermostat_setpoint == 19.0
    assert result.rows[1].thermostat_setpoint == 21.0
    assert result.rows[0].first_predicted_thermal_output == 0.0
    assert result.rows[1].first_predicted_thermal_output is not None
    assert result.rows[1].first_predicted_thermal_output > result.rows[0].first_predicted_thermal_output
