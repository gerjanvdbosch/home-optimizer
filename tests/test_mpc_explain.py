from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.features.mpc import (
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    MpcPlan,
    MpcPlanStep,
    explain_heating_plan,
)


def test_explain_heating_plan_reports_future_comfort_min_violation() -> None:
    start_time = datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=0.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.5,
            occupied=0.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
        )
        for step in range(6)
    ]
    plan = MpcPlan(
        status="ok",
        termination_condition="optimal",
        feasible=True,
        steps=[
            MpcPlanStep(
                timestamp_utc=step.timestamp_utc,
                hp_on=(index >= 1),
                start=(index == 1),
                stop=False,
                predicted_room_temp_c=19.5,
                temp_min_c=step.temp_min_c,
                temp_max_c=step.temp_max_c,
            )
            for index, step in enumerate(horizon)
        ],
    )

    explanation = explain_heating_plan(
        plan=plan,
        control_model=Rc2StateThermalControlModel(
            a11=0.99,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.01,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.0,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.2, mass_temp_c=19.2, hp_on=False, off_steps=1),
        horizon=horizon,
    )

    assert explanation is not None
    assert "Heating scheduled to prevent comfort-min violation at" in explanation
    assert "No-heat rollout reaches" in explanation
