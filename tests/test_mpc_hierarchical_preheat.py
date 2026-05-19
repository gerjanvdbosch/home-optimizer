from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.features.mpc import (
    MpcConstraints,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    PreheatBlock,
    SpaceHeatingFlexibilityAssessor,
    SpaceHeatingMpcControllerService,
    SpaceHeatingPreheatScheduler,
    ThermalFlexibilityState,
    ThermalFlexibilityStep,
)
from home_optimizer.features.mpc.preheat_scheduler import _PlanningState


def test_flexibility_assessor_sets_economic_target_above_temp_min() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=7.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.2,
            occupied=0.0,
            target_temp_c=20.5,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.22,
            export_price_eur_kwh=0.05,
        )
        for step in range(4)
    ]

    flexibility = SpaceHeatingFlexibilityAssessor().assess(
        interval_minutes=15,
        control_model=Rc2StateThermalControlModel(
            a11=0.98,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.35,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.9, mass_temp_c=19.9, hp_on=False, off_steps=1),
        horizon=horizon,
        constraints=MpcConstraints(),
    )

    assert flexibility.steps
    assert flexibility.steps[0].economic_target_c > horizon[0].temp_min_c
    assert flexibility.steps[0].economic_target_c < horizon[0].target_temp_c


def test_hierarchical_scheduler_ignores_single_spike() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flexibility = ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.6,
                no_heat_room_temp_c=19.7,
                no_heat_mass_temp_c=19.8,
                comfort_headroom_c=1.0,
                available_storage_kwh=1.2,
                expected_discharge_need_kwh=0.8,
                pv_surplus_forecast_kw=2.0 if step == 2 else 0.0,
                pv_surplus_window_kwh=0.33 if step == 2 else 0.0,
            )
            for step in range(6)
        ]
    )

    schedule = SpaceHeatingPreheatScheduler().build_schedule(
        flexibility_state=flexibility,
        constraints=MpcConstraints(min_on_steps=0, min_off_steps=0),
        interval_minutes=10,
    )

    assert schedule.blocks == []


def test_hierarchical_scheduler_builds_single_block_for_sustained_surplus() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flexibility = ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.6,
                no_heat_room_temp_c=19.5,
                no_heat_mass_temp_c=19.7,
                comfort_headroom_c=1.0,
                available_storage_kwh=2.0,
                expected_discharge_need_kwh=1.6,
                pv_surplus_forecast_kw=2.0 if 1 <= step <= 4 else 0.0,
                pv_surplus_window_kwh=1.3 if 1 <= step <= 4 else 0.0,
            )
            for step in range(8)
        ]
    )

    schedule = SpaceHeatingPreheatScheduler().build_schedule(
        flexibility_state=flexibility,
        constraints=MpcConstraints(),
        interval_minutes=10,
    )

    assert len(schedule.blocks) == 1
    assert schedule.blocks[0].max_starts == 1
    assert schedule.blocks[0].planned_charge_kwh > 0.0


def test_hierarchical_scheduler_merges_short_gap_inside_surplus_window() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flexibility = ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.8,
                no_heat_room_temp_c=19.4,
                no_heat_mass_temp_c=19.6,
                comfort_headroom_c=1.2,
                available_storage_kwh=2.5,
                expected_discharge_need_kwh=1.8,
                pv_surplus_forecast_kw=2.0 if step in {1, 2, 4, 5} else 0.0,
                pv_surplus_window_kwh=1.2 if step in {1, 2, 4, 5} else 0.0,
            )
            for step in range(8)
        ]
    )

    schedule = SpaceHeatingPreheatScheduler().build_schedule(
        flexibility_state=flexibility,
        constraints=MpcConstraints(),
        interval_minutes=10,
    )

    assert len(schedule.blocks) == 1
    assert schedule.blocks[0].start_index == 1
    assert schedule.blocks[0].end_index == 5


def test_hierarchical_scheduler_skips_long_low_energy_block() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flexibility = ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.8,
                no_heat_room_temp_c=19.4,
                no_heat_mass_temp_c=19.6,
                comfort_headroom_c=1.2,
                available_storage_kwh=0.02,
                expected_discharge_need_kwh=0.02,
                pv_surplus_forecast_kw=0.25 if 1 <= step <= 5 else 0.0,
                pv_surplus_window_kwh=0.2 if 1 <= step <= 5 else 0.0,
            )
            for step in range(8)
        ]
    )

    schedule = SpaceHeatingPreheatScheduler().build_schedule(
        flexibility_state=flexibility,
        constraints=MpcConstraints(),
        interval_minutes=10,
    )

    assert schedule.blocks == []


def test_hierarchical_scheduler_selects_best_block_under_required_charge_budget() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flexibility = ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.8,
                no_heat_room_temp_c=19.4,
                no_heat_mass_temp_c=19.6,
                comfort_headroom_c=0.6 if step < 4 else 1.6,
                available_storage_kwh=0.4 if step < 4 else 1.4,
                expected_discharge_need_kwh=0.4 if step < 4 else 1.0,
                pv_surplus_forecast_kw=(
                    1.0
                    if step in {1, 2}
                    else 2.0
                    if step in {10, 11, 12, 13}
                    else 0.0
                ),
                pv_surplus_window_kwh=(
                    0.3
                    if step in {1, 2}
                    else 1.6
                    if step in {10, 11, 12, 13}
                    else 0.0
                ),
            )
            for step in range(14)
        ],
        total_available_storage_kwh=1.4,
        total_expected_discharge_need_kwh=1.0,
    )

    schedule = SpaceHeatingPreheatScheduler().build_schedule(
        flexibility_state=flexibility,
        constraints=MpcConstraints(),
        interval_minutes=10,
    )

    assert len(schedule.blocks) == 1
    assert schedule.blocks[0].start_index == 10
    assert schedule.blocks[0].end_index == 13


def test_hierarchical_mode_limits_to_single_start_within_block() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=7.5,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.8 if 1 <= step <= 5 else 0.0,
            base_load_power_forecast_kw=0.3,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.24,
            export_price_eur_kwh=0.05,
        )
        for step in range(10)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
        ),
        control_model=Rc2StateThermalControlModel(
            a11=0.97,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.45,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.3, mass_temp_c=19.3, hp_on=False, off_steps=6),
    )

    assert plan.feasible is True
    block_starts = [step for step in plan.steps if step.preheat_block_id is not None and step.start]
    assert len(block_starts) <= 1
    assert any(step.preheat_block_id is not None for step in plan.steps)


def test_hierarchical_mode_allows_comfort_start_outside_preheat_block() -> None:
    start_time = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=5.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.6,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.30,
            export_price_eur_kwh=0.05,
        )
        for step in range(6)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
        ),
        control_model=Rc2StateThermalControlModel(
            a11=0.95,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            b_out_room=0.0,
            b_out_mass=0.0,
            b_solar_direct_room=0.0,
            b_heat_room=0.45,
            b_heat_mass=0.0,
            b_occ_room=0.0,
),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.05, mass_temp_c=19.05, hp_on=False, off_steps=6),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is True
    assert plan.steps[0].preheat_block_id is None


def test_model_based_run_selection_does_not_anchor_to_peak_pv_step() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=4.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=1.0 if step < 4 else 3.0 if step == 4 else 0.0,
            base_load_power_forecast_kw=0.2,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.30,
            export_price_eur_kwh=0.05,
        )
        for step in range(10)
    ]
    scheduler = SpaceHeatingPreheatScheduler()
    control_model = Rc2StateThermalControlModel(
        a11=0.94,
        a12=0.0,
        a21=0.0,
        a22=1.0,
        b_out_room=0.0,
        b_out_mass=0.0,
        b_solar_direct_room=0.0,
        b_heat_room=0.55,
        b_heat_mass=0.0,
        b_occ_room=0.0,
)
    block = PreheatBlock(
        block_id=0,
        start_index=0,
        end_index=4,
        start_time_utc=horizon[0].timestamp_utc,
        end_time_utc=horizon[4].timestamp_utc,
        available_surplus_kwh=1.4,
        available_storage_kwh=2.0,
        planned_charge_kwh=1.4,
        max_starts=1,
        min_run_steps=2,
        max_preheat_target_c=21.0,
        step_count=5,
    )

    evaluation = scheduler._select_best_run_in_block(
        control_model=control_model,
        state=_PlanningState(
            room_temp_c=19.1,
            mass_temp_c=None,
            q_heat_eff_kw=0.0,
            hp_on=False,
        ),
        horizon=horizon,
        block=block,
        interval_minutes=10,
        min_run_steps=2,
        min_off_steps=1,
        planned_charge_limit_kwh=1.4,
    )

    assert evaluation is not None
    assert evaluation.run_start_index < 4
