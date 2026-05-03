from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.domain import IdentifiedModel, NumericPoint, NumericSeries
from home_optimizer.features.mpc.control_oriented import (
    StateSpaceThermalModel,
    StateSpaceThermalMpcPlanRequest,
    StateSpaceThermalMpcService,
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


def disturbance_series(name: str, value: float) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=None,
        points=[
            NumericPoint(timestamp="2026-05-03T10:15:00+00:00", value=value),
            NumericPoint(timestamp="2026-05-03T10:30:00+00:00", value=value),
        ],
    )


def test_control_oriented_mpc_optimizes_for_comfort() -> None:
    service = StateSpaceThermalMpcService()
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())

    result = service.optimize(
        model=model,
        request=StateSpaceThermalMpcPlanRequest(
            start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
            initial_state=StateSpaceThermalState(room_temperature=19.0, floor_heat_state=0.0),
            allowed_thermal_outputs=[0.0, 4.0],
            move_block_times=[datetime(2026, 5, 3, 10, 15, tzinfo=timezone.utc)],
            outdoor_temperature_series=disturbance_series("outdoor_temperature", 10.0),
            solar_gain_series=disturbance_series("gti_living_room_windows_adjusted", 0.0),
            comfort_min_temperature=19.5,
            comfort_max_temperature=21.0,
            thermal_output_usage_penalty=0.0,
            thermal_output_change_penalty=0.0,
        ),
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert len(result.plan_results) == 1
    assert [point.value for point in result.recommended_plan.thermal_output_schedule.points] == [4.0, 4.0]


def test_control_oriented_mpc_respects_usage_and_change_penalties() -> None:
    service = StateSpaceThermalMpcService()
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())

    result = service.optimize(
        model=model,
        request=StateSpaceThermalMpcPlanRequest(
            start_time=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 3, 10, 30, tzinfo=timezone.utc),
            initial_state=StateSpaceThermalState(room_temperature=20.0, floor_heat_state=0.0),
            allowed_thermal_outputs=[0.0, 4.0],
            move_block_times=[datetime(2026, 5, 3, 10, 15, tzinfo=timezone.utc)],
            outdoor_temperature_series=disturbance_series("outdoor_temperature", 10.0),
            solar_gain_series=disturbance_series("gti_living_room_windows_adjusted", 0.0),
            comfort_min_temperature=19.0,
            comfort_max_temperature=21.0,
            thermal_output_usage_penalty=10.0,
            thermal_output_change_penalty=10.0,
            previous_applied_thermal_output=0.0,
        ),
    )

    assert [point.value for point in result.recommended_plan.thermal_output_schedule.points] == [0.0, 0.0]
    assert result.recommended_plan.thermal_output_change_cost == 0.0
