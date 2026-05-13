from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from home_optimizer.features.modeling import RoomRcConfig, RoomRcModel
from home_optimizer.features.modeling.room_2r2c import (
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)
from home_optimizer.features.modeling.room_arx import RoomArxConfig, RoomArxModel
from home_optimizer.features.mpc import (
    ControlModelConversionOptions,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    SpaceHeatingMpcControllerService,
    to_control_model,
)


def test_to_control_model_from_arx_aggregates_matching_coefficients() -> None:
    model = RoomArxModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomArxConfig(),
        feature_names=[
            "room_temperature_lag_0",
            "room_temperature_lag_1",
            "outdoor_temperature_lag_0",
            "thermal_output_lag_0",
            "solar_gain_lag_0",
            "occupied_flag_lag_0",
        ],
        intercept=1.25,
        coefficients=[0.6, 0.1, 0.05, 0.4, 0.02, 0.03],
        sample_count=100,
    )

    control_model = to_control_model(
        model,
        options=ControlModelConversionOptions(solar_gain_input_scale=0.5),
    )

    assert control_model.a == pytest.approx(0.7)
    assert control_model.b_out == pytest.approx(0.05)
    assert control_model.b_heat == pytest.approx(0.4)
    assert control_model.b_solar == pytest.approx(0.04)
    assert control_model.b_occ == pytest.approx(0.03)
    assert control_model.c == pytest.approx(1.25)


def test_to_control_model_from_room_rc_matches_reduced_discrete_terms() -> None:
    room_rc_model = RoomRcModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(),
        params=RoomRC2StateParams().to_dict(),
        sample_count=100,
    )

    control_model = to_control_model(room_rc_model)
    physical = RoomRC2StatePhysicalModel(RoomRcConfig(interval_minutes=10))
    f_matrix, g_matrix, a_discrete, _ = physical.params_to_matrices(RoomRC2StateParams())
    steady_state_gain = -np.linalg.inv(f_matrix) @ g_matrix
    dominant_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(a_discrete))))
    one_minus_a = 1.0 - dominant_eigenvalue

    assert control_model.a == pytest.approx(dominant_eigenvalue)
    assert control_model.b_out == pytest.approx(float(steady_state_gain[0, 0] * one_minus_a))
    assert control_model.b_heat == pytest.approx(float(steady_state_gain[0, 1] * one_minus_a))
    assert control_model.b_solar == pytest.approx(
        float((steady_state_gain[0, 2] + steady_state_gain[0, 3]) * one_minus_a)
    )
    assert control_model.b_occ == pytest.approx(float(steady_state_gain[0, 4] * one_minus_a))


def test_room_rc_control_model_allows_heating_to_raise_predicted_temperature() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    room_rc_model = RoomRcModel(
        trained_from_utc=start_time,
        trained_to_utc=start_time + timedelta(days=1),
        interval_minutes=15,
        config=RoomRcConfig(),
        params=RoomRC2StateParams().to_dict(),
        sample_count=100,
    )

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=[
                MpcHorizonStep(
                    timestamp_utc=start_time + timedelta(minutes=15 * step),
                    outdoor_temp_c=10.0,
                    solar_gain_kw=0.0,
                    effective_heating_kw_forecast=2.5,
                    occupied=0.0,
                    temp_min_c=20.0,
                    temp_max_c=21.0,
                )
                for step in range(12)
            ],
        ),
        control_model=to_control_model(room_rc_model),
        initial_state=MpcInitialState(room_temp_c=18.0, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    heated_steps = [step for step in plan.steps if step.hp_on]
    assert heated_steps
    assert max(
        step.predicted_room_temp_c for step in heated_steps
    ) > heated_steps[0].predicted_room_temp_c
