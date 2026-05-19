from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.features.backtest.runner import SpaceHeatingMpcBacktestRunner
from home_optimizer.features.mpc import (
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveBreakdown,
    MpcPlan,
    MpcPlanStep,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    Rc2StateThermalControlModel,
)


class _StaticPlanController:
    def plan(self, request: MpcControllerRequest, **kwargs) -> MpcPlan:
        step = request.horizon[0]
        return MpcPlan(
            status="ok",
            termination_condition="optimal",
            feasible=True,
            objective_breakdown=MpcObjectiveBreakdown(
                comfort_low=1.0,
                active_comfort_high=2.0,
                tracking_under_target=1.0,
                tracking_over_target=2.0,
                terminal=4.0,
                start=5.0,
                runtime=6.0,
                energy_cost=7.0,
            ),
            steps=[
                MpcPlanStep(
                    timestamp_utc=step.timestamp_utc,
                    hp_on=False,
                    start=False,
                    stop=False,
                    predicted_room_temp_c=0.0,
                    temp_min_c=step.temp_min_c,
                    temp_max_c=step.temp_max_c,
                    slack_low_c=0.0,
                    slack_high_c=0.0,
                    effective_heating_kw=0.0,
                    price_eur_kwh=step.import_price_eur_kwh,
                    estimated_energy_cost_eur=0.0,
                )
            ],
        )


class _ReplayProvider:
    def __init__(self, horizon: list[MpcHorizonStep]) -> None:
        self.horizon = horizon

    def get_forecast_horizon(self, *args, **kwargs):
        class _ReplayHorizon:
            def __init__(self, horizon: list[MpcHorizonStep]) -> None:
                self.horizon = horizon
                self.forecast_issue_time_utc = horizon[0].timestamp_utc
                self.forecast_age_minutes = 0.0
                self.forecast_coverage_ratio = 1.0
                self.missing_forecast_count = 0

        return _ReplayHorizon(self.horizon)


def _step(
    hour: int,
    *,
    realized_room_temp_c: float,
) -> MpcHorizonStep:
    return MpcHorizonStep(
        timestamp_utc=datetime(2026, 5, 14, hour, 0, tzinfo=timezone.utc),
        outdoor_temp_c=0.0,
        solar_gain_kw=0.0,
        effective_heating_kw_forecast=0.0,
        hp_electric_power_forecast_kw=0.0,
        pv_available_power_forecast_kw=0.0,
        base_load_power_forecast_kw=0.0,
        occupied=0.0,
        temp_min_c=19.0,
        temp_max_c=21.0,
        import_price_eur_kwh=0.25,
        export_price_eur_kwh=0.0,
        realized_room_temp_c=realized_room_temp_c,
    )


def test_backtest_runner_keeps_simulated_state_instead_of_resetting_to_historical_room_temperature() -> None:
    runner = SpaceHeatingMpcBacktestRunner(controller=_StaticPlanController())
    result = runner.run(
        model_id="room-model-v1",
        model_type="room_2r2c",
        control_model=Rc2StateThermalControlModel(
            a11=0.5,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.0,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        timeline=[
            _step(0, realized_room_temp_c=20.0),
            _step(1, realized_room_temp_c=50.0),
            _step(2, realized_room_temp_c=60.0),
        ],
        initial_state=Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0, hp_on=False, on_steps=0, off_steps=1),
        interval_minutes=10,
        horizon_steps=2,
    )

    assert [step.simulated_next_room_temp_c for step in result.step_results] == [10.0, 5.0]
    assert [step.historical_next_room_temp_c for step in result.step_results] == [50.0, 60.0]
    assert result.mpc_objective_breakdown.total == 0.0
    assert result.solver_objective_breakdown.total == 56.0
    assert result.solver_objective_breakdown.temperature_tracking == 6.0


def test_backtest_runner_advances_2state_mass_state_across_steps() -> None:
    runner = SpaceHeatingMpcBacktestRunner(controller=_StaticPlanController())
    result = runner.run(
        model_id="room-rc-v1",
        model_type="room_2r2c",
        control_model=Rc2StateThermalControlModel(
            a11=0.0,
            a12=1.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_heat_room=0.0,
            b_heat_mass=0.0,
            b_solar_direct_room=0.0,
            b_solar_filtered_room=0.0,
            b_solar_direct_mass=0.0,
            b_solar_filtered_mass=0.0,
            b_occ_room=0.0,
            b_occ_mass=0.0,
            c_room=0.0,
            c_mass=1.0,
        ),
        timeline=[
            _step(0, realized_room_temp_c=20.0),
            _step(1, realized_room_temp_c=21.0),
            _step(2, realized_room_temp_c=22.0),
        ],
        initial_state=Rc2StateMpcInitialState(
            room_temp_c=20.0,
            mass_temp_c=30.0,
            hp_on=False,
            on_steps=0,
            off_steps=1,
        ),
        interval_minutes=10,
        horizon_steps=2,
    )

    assert [step.simulated_next_room_temp_c for step in result.step_results] == [30.0, 31.0]


def test_backtest_runner_uses_realized_exogenous_for_plant_in_forecast_replay() -> None:
    runner = SpaceHeatingMpcBacktestRunner(controller=_StaticPlanController())
    realized_timeline = [
        MpcHorizonStep(
            timestamp_utc=datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
            outdoor_temp_c=0.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=0.0,
            hp_electric_power_forecast_kw=1.0,
            pv_available_power_forecast_kw=0.0,
            pv_available_power_realized_kw=0.0,
            base_load_power_forecast_kw=0.0,
            base_load_power_realized_kw=0.0,
            occupied=0.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.0,
            realized_room_temp_c=20.0,
        ),
        MpcHorizonStep(
            timestamp_utc=datetime(2026, 5, 14, 12, 10, tzinfo=timezone.utc),
            outdoor_temp_c=0.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=0.0,
            hp_electric_power_forecast_kw=1.0,
            pv_available_power_forecast_kw=0.0,
            pv_available_power_realized_kw=0.0,
            base_load_power_forecast_kw=0.0,
            base_load_power_realized_kw=0.0,
            occupied=0.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.0,
            realized_room_temp_c=20.0,
        ),
    ]
    replay_horizon = [
        realized_timeline[0].model_copy(
            update={
                "solar_gain_kw": 100.0,
                "solar_gain_mass_kw": 100.0,
                "solar_irradiance_forecast_w_m2": 1000.0,
                "pv_available_power_forecast_kw": 10.0,
            }
        )
    ]

    result = runner.run(
        model_id="room-model-v1",
        model_type="room_2r2c",
        control_model=Rc2StateThermalControlModel(
            a11=1.0,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=1.0,
            b_heat_room=0.0,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        timeline=realized_timeline,
        initial_state=Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0, hp_on=False, on_steps=0, off_steps=1),
        interval_minutes=10,
        horizon_steps=1,
        exogenous_mode="forecast_replay",
        forecast_replay_provider=_ReplayProvider(replay_horizon),
    )

    assert result.step_results[0].pv_forecast_kw == pytest.approx(10.0)
    assert result.step_results[0].pv_realized_kw == pytest.approx(0.0)
    assert result.step_results[0].simulated_next_room_temp_c == pytest.approx(20.0)


def test_backtest_runner_computes_pv_surplus_capture_and_safe_zero_ratio() -> None:
    runner = SpaceHeatingMpcBacktestRunner(controller=_StaticPlanController())
    result = runner.run(
        model_id="room-model-v1",
        model_type="room_2r2c",
        control_model=Rc2StateThermalControlModel(
            a11=1.0,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.0,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        timeline=[
            MpcHorizonStep(
                timestamp_utc=datetime(2026, 5, 14, 0, 0, tzinfo=timezone.utc),
                outdoor_temp_c=0.0,
                solar_gain_kw=0.0,
                solar_irradiance_forecast_w_m2=100.0,
                solar_irradiance_realized_w_m2=100.0,
                effective_heating_kw_forecast=0.0,
                hp_electric_power_forecast_kw=2.0,
                pv_available_power_forecast_kw=3.0,
                pv_available_power_realized_kw=3.0,
                base_load_power_forecast_kw=1.0,
                base_load_power_realized_kw=1.0,
                occupied=0.0,
                temp_min_c=19.0,
                temp_max_c=21.0,
                import_price_eur_kwh=0.25,
                export_price_eur_kwh=0.0,
                realized_room_temp_c=20.0,
            ),
            MpcHorizonStep(
                timestamp_utc=datetime(2026, 5, 14, 0, 10, tzinfo=timezone.utc),
                outdoor_temp_c=0.0,
                solar_gain_kw=0.0,
                solar_irradiance_forecast_w_m2=0.0,
                solar_irradiance_realized_w_m2=0.0,
                effective_heating_kw_forecast=0.0,
                hp_electric_power_forecast_kw=2.0,
                pv_available_power_forecast_kw=0.0,
                pv_available_power_realized_kw=0.0,
                base_load_power_forecast_kw=1.0,
                base_load_power_realized_kw=1.0,
                occupied=0.0,
                temp_min_c=19.0,
                temp_max_c=21.0,
                import_price_eur_kwh=0.25,
                export_price_eur_kwh=0.0,
                realized_room_temp_c=20.0,
            ),
        ],
        initial_state=Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0, hp_on=False, on_steps=0, off_steps=1),
        interval_minutes=10,
        horizon_steps=1,
    )

    assert result.pv_diagnostics.realized_pv_surplus_kwh == pytest.approx(2.0 * (10.0 / 60.0))
    assert result.pv_diagnostics.forecast_pv_surplus_kwh == pytest.approx(2.0 * (10.0 / 60.0))
    assert result.pv_diagnostics.mpc_hp_energy_kwh == pytest.approx(0.0)
    assert result.pv_diagnostics.mpc_realized_pv_surplus_capture_ratio == pytest.approx(0.0)
    assert result.pv_diagnostics.mpc_forecast_pv_surplus_capture_ratio == pytest.approx(0.0)
