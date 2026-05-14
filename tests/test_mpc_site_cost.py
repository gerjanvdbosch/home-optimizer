from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.features.mpc import (
    LinearThermalControlModel,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    SpaceHeatingMpcControllerService,
)


def test_site_cost_prefers_pv_surplus_window_for_heating() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0 if step == 0 else 3.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            temp_min_c=20.0,
            temp_max_c=24.0,
            import_price_eur_kwh=0.30,
            export_price_eur_kwh=0.05,
        )
        for step in range(4)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=15, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.99,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.3,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.3, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False
    assert plan.steps[1].hp_on is True
    assert plan.steps[1].estimated_energy_cost_eur == 0.025
    assert plan.steps[1].estimated_energy_cost_eur < (0.30 * 2.0 * 0.25)
