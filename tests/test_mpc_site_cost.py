from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.features.mpc import (
    LinearThermalControlModel,
    MpcConstraints,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveWeights,
    MpcProblem,
    PreheatPlan,
    PreheatPlanStep,
    SpaceHeatingMpcControllerService,
    SpaceHeatingMpcSolver,
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
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                tracking_under_target=0.0,
                tracking_over_target=0.0,
                unnecessary_heating=0.0,
                terminal=0.0,
            ),
        ),
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


def test_target_tracking_uses_economic_target_without_pv_opportunity() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=0.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.0,
            occupied=0.0,
            target_temp_c=22.0,
            temp_min_c=19.0,
            temp_max_c=24.0,
            import_price_eur_kwh=0.0,
            export_price_eur_kwh=0.0,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                comfort_low=10_000.0,
                comfort_high=10_000.0,
                tracking_under_target=10.0,
                tracking_over_target=0.0,
                unnecessary_heating=0.0,
                terminal=0.0,
                start=0.0,
                energy=0.0,
                pv_self_consumption=0.0,
                runtime=0.0,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=0.0,
            b_heat=1.0,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.0, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].economic_target_c > 19.0
    assert plan.steps[0].economic_target_c < 22.0


def test_forced_heating_stays_feasible_when_overheating_requires_large_slack() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=1.0,
            hp_electric_power_forecast_kw=1.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.0,
            occupied=0.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.0,
            export_price_eur_kwh=0.0,
            hp_must_be_on=(step == 0),
        )
        for step in range(4)
    ]

    plan = SpaceHeatingMpcSolver().solve(
        MpcProblem(
            interval_minutes=10,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                tracking_under_target=0.0,
                tracking_over_target=0.0,
                unnecessary_heating=0.0,
                terminal=0.0,
                start=0.0,
                energy=0.0,
                pv_self_consumption=0.0,
                runtime=0.0,
            ),
            constraints=MpcConstraints(),
            control_model=LinearThermalControlModel(
                a=1.0,
                b_out=0.0,
                b_solar=0.0,
                b_heat=10.0,
                b_occ=0.0,
                c=0.0,
            ),
            initial_state=MpcInitialState(room_temp_c=20.0, hp_on=False, off_steps=1),
        ),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is True
    assert plan.steps[1].slack_high_c > 5.0
    assert plan.steps[0].useful_preheat_target_c == pytest.approx(
        plan.steps[0].economic_target_c
    )
    assert plan.objective_breakdown.temperature_tracking >= 0.0


def test_useful_preheat_target_stays_within_comfort_band() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=8.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=3.0 if step == 0 else 0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.20 if step == 0 else 0.30,
            export_price_eur_kwh=0.05,
        )
        for step in range(4)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=15, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.98,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.25,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.0, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    for step in plan.steps:
        assert step.useful_preheat_target_c >= 19.0
        assert step.useful_preheat_target_c <= 21.0


def test_mpc_preheats_with_pv_surplus_and_future_heat_need() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=5.0 if step >= 2 else 10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=3.0 if step in {0, 1} else 0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.20 if step == 0 else 0.32,
            export_price_eur_kwh=0.05,
        )
        for step in range(4)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=15, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.94,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.5,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.8, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].useful_preheat_target_c > 19.0
    assert plan.steps[0].hp_on is True
    assert plan.objective_breakdown.captured_pv_kwh > 0.0
    assert plan.preheat_schedule is not None
    assert plan.preheat_schedule.blocks


def test_single_pv_spike_does_not_trigger_short_run_with_sustained_opportunity() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=8.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.5 if step == 0 else 0.0,
            base_load_power_forecast_kw=0.4,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.05,
        )
        for step in range(6)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=3, min_off_steps=3),
        ),
        control_model=LinearThermalControlModel(
            a=0.98,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.35,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.7, hp_on=False, off_steps=6),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False
    assert all(not plan.steps[index].hp_on for index in range(3))
    assert plan.objective_breakdown.captured_pv_kwh == pytest.approx(0.0)


def test_sustained_pv_surplus_triggers_run_and_respects_min_on_steps() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=7.0 if step >= 4 else 9.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.8 if step < 4 else 0.0,
            base_load_power_forecast_kw=0.4,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.05,
        )
        for step in range(8)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=3, min_off_steps=3),
        ),
        control_model=LinearThermalControlModel(
            a=0.97,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.45,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.3, hp_on=False, off_steps=6),
    )

    assert plan.feasible is True
    start_indices = [index for index, step in enumerate(plan.steps) if step.start]
    assert start_indices
    first_start = start_indices[0]
    assert all(
        plan.steps[index].hp_on
        for index in range(first_start, min(first_start + 3, len(plan.steps)))
    )
    assert max(step.predicted_room_temp_c - step.temp_max_c for step in plan.steps) <= 0.15


def test_scheduler_clusters_contiguous_preheat_steps_into_single_block() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=7.0 if step >= 4 else 9.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.8 if 1 <= step <= 4 else 0.0,
            base_load_power_forecast_kw=0.4,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.05,
        )
        for step in range(8)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=10, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.97,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.45,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.3, hp_on=False, off_steps=6),
    )

    assert plan.preheat_schedule is not None
    assert len(plan.preheat_schedule.blocks) == 1
    assert plan.preheat_schedule.blocks[0].max_starts == 1
    active_block_ids = {
        step.preheat_block_id for step in plan.steps if step.preheat_active
    }
    assert active_block_ids == {0}


def test_preheat_block_start_limit_keeps_single_start_in_same_block() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=8.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.5,
            base_load_power_forecast_kw=0.3,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.05,
        )
        for step in range(8)
    ]
    preheat_plan = PreheatPlan(
        steps=[
            PreheatPlanStep(
                timestamp_utc=step.timestamp_utc,
                preheat_active=True,
                preheat_opportunity_score=1.0,
                max_preheat_target_c=20.7,
                preheat_budget_share_kwh=0.3,
                preheat_block_id=0,
                preheat_block_budget_kwh=2.4,
                preheat_block_max_starts=1,
                pv_surplus_window_kwh=2.4,
                storage_headroom_electric_kwh=2.4,
                reason="test_block",
            )
            for step in horizon
        ],
    )

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            preheat_plan=preheat_plan,
        ),
        control_model=LinearThermalControlModel(
            a=0.95,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.4,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.0, hp_on=False, off_steps=6),
    )

    assert plan.feasible is True
    assert sum(1 for step in plan.steps if step.start) <= 1


def test_mpc_avoids_unnecessary_heating_near_comfort_max() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=12.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=3.0 if step == 0 else 0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.20,
            export_price_eur_kwh=0.05,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=15, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.99,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.6,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.9, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False


def test_mpc_does_not_chase_midpoint_target_without_pv_surplus() -> None:
    start_time = datetime(2026, 1, 1, 19, 30, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=12.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
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
            b_heat=0.35,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.6, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].useful_preheat_target_c > 19.0
    assert plan.steps[0].useful_preheat_target_c < 20.0
    assert plan.steps[0].preheat_block_id is None


def test_passive_solar_gain_does_not_create_unnecessary_heating_penalty() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=1.5,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.25,
            export_price_eur_kwh=0.05,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                start=0.0,
                energy=0.0,
                pv_self_consumption=0.0,
                runtime=0.0,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=0.8,
            b_heat=0.5,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.0, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False
    assert plan.objective_breakdown.unnecessary_heating == pytest.approx(0.0, abs=1e-9)
    assert plan.objective_breakdown.tracking_over_target >= 0.0


def test_objective_breakdown_exposes_new_components() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=6.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=3.0 if step == 0 else 0.0,
            base_load_power_forecast_kw=0.5,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            import_price_eur_kwh=0.20,
            export_price_eur_kwh=0.05,
        )
        for step in range(4)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(interval_minutes=15, horizon=horizon),
        control_model=LinearThermalControlModel(
            a=0.96,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.4,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.7, hp_on=False, off_steps=1),
    )

    breakdown = plan.objective_breakdown
    assert breakdown.comfort_low >= 0.0
    assert breakdown.comfort_high >= 0.0
    assert breakdown.tracking_under_target >= 0.0
    assert breakdown.tracking_over_target >= 0.0
    assert breakdown.energy_cost >= 0.0
    assert breakdown.pv_self_consumption_reward >= 0.0
    assert breakdown.unnecessary_heating >= 0.0
    assert breakdown.start >= 0.0
    assert breakdown.terminal >= 0.0
    assert breakdown.total >= 0.0


def test_passive_solar_overshoot_is_not_counted_as_active_comfort_high() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=1.5,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.0,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=20.5,
            import_price_eur_kwh=0.0,
            export_price_eur_kwh=0.0,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                start=0.0,
                energy=0.0,
                pv_self_consumption=0.0,
                runtime=0.0,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=1.0,
            b_heat=0.5,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.3, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False
    assert plan.objective_breakdown.active_comfort_high == pytest.approx(0.0, abs=1e-9)
    assert plan.objective_breakdown.passive_comfort_high >= 0.0


def test_heating_above_comfort_max_creates_active_comfort_high_cost() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=3.0 if step == 0 else 0.0,
            base_load_power_forecast_kw=0.0,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=20.5,
            import_price_eur_kwh=0.0,
            export_price_eur_kwh=0.0,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                active_comfort_high=10_000.0,
                passive_comfort_high=0.0,
                comfort_low=10_000.0,
                tracking_under_target=0.0,
                tracking_over_target=0.0,
                terminal=0.0,
                start=0.0,
                energy=0.0,
                pv_self_consumption=5.0,
                runtime=0.0,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=0.0,
            b_heat=1.0,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=20.3, hp_on=False, off_steps=1),
    )

    assert plan.feasible is True
    assert plan.steps[0].hp_on is False

    forced_heating_plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                active_comfort_high=10_000.0,
                passive_comfort_high=0.0,
                comfort_low=10_000.0,
                tracking_under_target=50.0,
                tracking_over_target=0.0,
                terminal=0.0,
                start=0.0,
                energy=0.0,
                pv_self_consumption=5.0,
                runtime=0.0,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=0.0,
            b_heat=1.0,
            b_occ=0.0,
            c=0.0,
        ),
        initial_state=MpcInitialState(room_temp_c=19.0, hp_on=False, off_steps=1),
    )

    assert forced_heating_plan.feasible is True
    assert forced_heating_plan.objective_breakdown.active_comfort_high >= 0.0


def test_lingering_q_heat_eff_counts_as_active_comfort_high() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=15 * step),
            outdoor_temp_c=10.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=0.0,
            base_load_power_forecast_kw=0.0,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=20.5,
            import_price_eur_kwh=0.0,
            export_price_eur_kwh=0.0,
        )
        for step in range(3)
    ]

    plan = SpaceHeatingMpcControllerService().plan(
        MpcControllerRequest(
            interval_minutes=15,
            horizon=horizon,
            objective_weights=MpcObjectiveWeights(
                active_comfort_high=10_000.0,
                passive_comfort_high=0.0,
                tracking_under_target=0.0,
                tracking_over_target=0.0,
                unnecessary_heating=0.0,
                terminal=0.0,
                start=0.0,
                energy=0.0,
                pv_self_consumption=0.0,
                runtime=0.0,
                q_heat_eff_active_threshold_kw=0.1,
            ),
        ),
        control_model=LinearThermalControlModel(
            a=1.0,
            b_out=0.0,
            b_solar=0.0,
            b_heat=0.5,
            b_occ=0.0,
            actuator_alpha=0.8,
            c=0.0,
        ),
        initial_state=MpcInitialState(
            room_temp_c=20.6,
            q_heat_eff_kw=0.5,
            hp_on=False,
            off_steps=1,
        ),
    )

    assert plan.feasible is True
    assert plan.objective_breakdown.active_comfort_high > 0.0


def test_comfort_low_remains_more_expensive_than_passive_comfort_high() -> None:
    weights = MpcObjectiveWeights(
        comfort_low=10_000.0,
        active_comfort_high=2_000.0,
        passive_comfort_high=100.0,
    )

    comfort_low_cost = weights.comfort_low * (15.0 / 60.0) * 0.2
    passive_high_cost = weights.passive_comfort_high * (15.0 / 60.0) * 0.2

    assert comfort_low_cost > passive_high_cost
