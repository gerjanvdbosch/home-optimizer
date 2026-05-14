from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.features.mpc import (
    LinearThermalControlModel,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcPlan,
    MpcPlanStep,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    SpaceHeatingMpcBacktestRunner,
)


class _StaticPlanController:
    def plan(self, request: MpcControllerRequest, **kwargs) -> MpcPlan:
        step = request.horizon[0]
        return MpcPlan(
            status="ok",
            termination_condition="optimal",
            feasible=True,
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
        model_type="room_arx",
        control_model=LinearThermalControlModel(
            a=0.5,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.0,
            b_occ=0.0,
            c=0.0,
        ),
        timeline=[
            _step(0, realized_room_temp_c=20.0),
            _step(1, realized_room_temp_c=50.0),
            _step(2, realized_room_temp_c=60.0),
        ],
        initial_state=MpcInitialState(room_temp_c=20.0, hp_on=False, on_steps=0, off_steps=1),
        interval_minutes=10,
        horizon_steps=2,
    )

    assert [step.simulated_next_room_temp_c for step in result.step_results] == [10.0, 5.0]
    assert [step.historical_next_room_temp_c for step in result.step_results] == [50.0, 60.0]


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
