from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.domain import IdentifiedModel
from home_optimizer.domain import NumericPoint, NumericSeries
from home_optimizer.features.mpc.control_oriented import (
    StateSpaceThermalModel,
    StateSpaceThermalPredictionRequest,
    StateSpaceThermalPredictionService,
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


def test_state_space_prediction_service_predicts_room_and_floor_series() -> None:
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())
    service = StateSpaceThermalPredictionService()

    result = service.predict(
        model=model,
        request=StateSpaceThermalPredictionRequest(
            start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
            initial_state=StateSpaceThermalState(room_temperature=20.0, floor_heat_state=1.0),
            thermal_output_schedule=NumericSeries(
                name="thermal_output",
                unit="kW",
                points=[
                    NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=4.0),
                    NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=0.0),
                ],
            ),
            outdoor_temperature_series=NumericSeries(
                name="outdoor_temperature",
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=10.0),
                    NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=10.0),
                ],
            ),
            solar_gain_series=NumericSeries(
                name="gti_living_room_windows_adjusted",
                unit="Wm2",
                points=[
                    NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=100.0),
                    NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=0.0),
                ],
            ),
        ),
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert [point.timestamp for point in result.room_temperature.points] == [
        "2026-05-03T10:15:00+00:00",
        "2026-05-03T10:30:00+00:00",
    ]
    assert [point.value for point in result.room_temperature.points] == pytest.approx(
        [19.345, 18.63915]
    )
    assert [point.value for point in result.floor_heat_state.points] == pytest.approx(
        [1.09, 1.0573]
    )


def test_state_space_prediction_service_requires_full_input_coverage() -> None:
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())
    service = StateSpaceThermalPredictionService()

    with pytest.raises(ValueError, match="missing state-space prediction input"):
        service.predict(
            model=model,
            request=StateSpaceThermalPredictionRequest(
                start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
                initial_state=StateSpaceThermalState(room_temperature=20.0, floor_heat_state=1.0),
                thermal_output_schedule=NumericSeries(
                    name="thermal_output",
                    unit="kW",
                    points=[],
                ),
                outdoor_temperature_series=NumericSeries(
                    name="outdoor_temperature",
                    unit="degC",
                    points=[
                        NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=10.0),
                        NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=10.0),
                    ],
                ),
                solar_gain_series=NumericSeries(
                    name="gti_living_room_windows_adjusted",
                    unit="Wm2",
                    points=[
                        NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=100.0),
                        NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=0.0),
                    ],
                ),
            ),
        )
