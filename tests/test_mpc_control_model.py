from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
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
    assert control_model.actuator_alpha == pytest.approx(0.0)
    assert control_model.c == pytest.approx(1.25)


def test_to_control_model_from_room_rc_defaults_to_exact_2state_conversion() -> None:
    params = RoomRC2StateParams()
    room_rc_model = RoomRcModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(),
        params=params.to_dict(),
        sample_count=100,
    )

    control_model = to_control_model(room_rc_model)

    assert isinstance(control_model, Rc2StateThermalControlModel)
    physical = RoomRC2StatePhysicalModel(RoomRcConfig(interval_minutes=10))
    _, _, a_discrete, b_discrete = physical.params_to_matrices(params)
    assert control_model.a11 == pytest.approx(float(a_discrete[0, 0]))
    assert control_model.a12 == pytest.approx(float(a_discrete[0, 1]))
    assert control_model.a21 == pytest.approx(float(a_discrete[1, 0]))
    assert control_model.a22 == pytest.approx(float(a_discrete[1, 1]))
    assert control_model.b_out_room == pytest.approx(float(b_discrete[0, 0]))
    assert control_model.b_out_mass == pytest.approx(float(b_discrete[1, 0]))
    assert control_model.b_heat_room == pytest.approx(float(b_discrete[0, 1]))
    assert control_model.b_heat_mass == pytest.approx(float(b_discrete[1, 1]))
    assert control_model.b_solar_direct_room == pytest.approx(float(b_discrete[0, 2]))
    assert control_model.b_solar_filtered_room == pytest.approx(float(b_discrete[0, 3]))
    assert control_model.b_solar_direct_mass == pytest.approx(float(b_discrete[1, 2]))
    assert control_model.b_solar_filtered_mass == pytest.approx(float(b_discrete[1, 3]))
    assert control_model.b_hour_sin_room == pytest.approx(float(b_discrete[0, 5]))
    assert control_model.b_hour_cos_room == pytest.approx(float(b_discrete[0, 6]))
    assert control_model.b_hour_sin_mass == pytest.approx(float(b_discrete[1, 5]))
    assert control_model.b_hour_cos_mass == pytest.approx(float(b_discrete[1, 6]))
    assert control_model.actuator_alpha == pytest.approx(room_rc_model.config.alpha_heat)


def test_room_rc_control_model_keeps_heating_active_under_comfort_pressure() -> None:
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
        initial_state=Rc2StateMpcInitialState(
            room_temp_c=18.0,
            mass_temp_c=18.0,
            hp_on=False,
            off_steps=1,
        ),
    )

    assert plan.feasible is True
    heated_steps = [step for step in plan.steps if step.hp_on]
    assert heated_steps
    assert heated_steps[0].start is True
    assert len(heated_steps) >= len(plan.steps) - 1


def test_room_rc_2state_control_model_allows_heating_to_raise_predicted_temperature() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    room_rc_model = RoomRcModel(
        trained_from_utc=start_time,
        trained_to_utc=start_time + timedelta(days=1),
        interval_minutes=10,
        config=RoomRcConfig(),
        params=RoomRC2StateParams().to_dict(),
        sample_count=100,
    )

    plan = SpaceHeatingMpcControllerService().plan_from_source_model(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=[
                MpcHorizonStep(
                    timestamp_utc=start_time + timedelta(minutes=10 * step),
                    outdoor_temp_c=5.0,
                    solar_gain_kw=0.0,
                    effective_heating_kw_forecast=2.5,
                    occupied=0.0,
                    temp_min_c=20.0,
                    temp_max_c=21.0,
                )
                for step in range(12)
            ],
        ),
        source_model=room_rc_model,
        initial_state=Rc2StateMpcInitialState(
            room_temp_c=18.0,
            mass_temp_c=18.0,
            hp_on=False,
            off_steps=1,
        ),
    )

    assert plan.feasible is True
    heated_steps = [step for step in plan.steps if step.hp_on]
    assert heated_steps
    assert heated_steps[0].start is True
