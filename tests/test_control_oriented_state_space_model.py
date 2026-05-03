from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.domain import IdentifiedModel
from home_optimizer.features.mpc.control_oriented import (
    StateSpaceThermalControlInput,
    StateSpaceThermalDisturbance,
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


def test_state_space_model_builds_from_identified_model() -> None:
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())

    assert model.model_name == "linear_2state_room_temperature"
    assert model.state_transition_matrix[0] == pytest.approx((0.9, 0.485))
    assert model.state_transition_matrix[1] == pytest.approx((0.0, 0.97))
    assert model.input_matrix[0] == pytest.approx((0.015,))
    assert model.input_matrix[1] == pytest.approx((0.03,))
    assert model.disturbance_matrix[0] == pytest.approx((0.02, 0.001))
    assert model.disturbance_matrix[1] == pytest.approx((0.0, 0.0))
    assert model.affine_offset == pytest.approx((0.5, 0.0))


def test_state_space_model_steps_forward_one_interval() -> None:
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())

    next_state = model.step(
        StateSpaceThermalState(room_temperature=20.0, floor_heat_state=1.0),
        control_input=StateSpaceThermalControlInput(thermal_output=4.0),
        disturbance=StateSpaceThermalDisturbance(
            outdoor_temperature=10.0,
            solar_gain=100.0,
        ),
    )

    assert next_state.room_temperature == pytest.approx(19.345)
    assert next_state.floor_heat_state == pytest.approx(1.09)


def test_state_space_model_simulates_multiple_steps() -> None:
    model = StateSpaceThermalModel.from_identified_model(build_room_temperature_model())

    states = model.simulate(
        initial_state=StateSpaceThermalState(room_temperature=20.0, floor_heat_state=1.0),
        control_inputs=[
            StateSpaceThermalControlInput(thermal_output=4.0),
            StateSpaceThermalControlInput(thermal_output=0.0),
        ],
        disturbances=[
            StateSpaceThermalDisturbance(outdoor_temperature=10.0, solar_gain=100.0),
            StateSpaceThermalDisturbance(outdoor_temperature=10.0, solar_gain=0.0),
        ],
    )

    assert [state.room_temperature for state in states] == pytest.approx([19.345, 18.63915])
    assert [state.floor_heat_state for state in states] == pytest.approx([1.09, 1.0573])


def test_state_space_model_validates_required_coefficients() -> None:
    identified_model = build_room_temperature_model().model_copy(
        update={
            "coefficients": {
                "previous_room_temperature": 0.9,
                "outdoor_temperature": 0.02,
            }
        }
    )

    with pytest.raises(ValueError, match="missing 2-state room-temperature coefficients"):
        StateSpaceThermalModel.from_identified_model(identified_model)
