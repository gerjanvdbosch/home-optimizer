from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.features.mpc import (
    ExecutionTargetStep,
    HeatPumpSequencer,
    HeatPumpSequencerState,
    MpcConstraints,
    MpcHorizonStep,
    PreheatBlock,
    PreheatSchedule,
    ThermalFlexibilityState,
    ThermalFlexibilityStep,
)


def _build_horizon(start_time: datetime, count: int, *, block_id: int | None) -> list[MpcHorizonStep]:
    return [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=10 * step),
            outdoor_temp_c=7.0,
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=2.5 if block_id is not None else 0.0,
            base_load_power_forecast_kw=0.3,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=19.0,
            temp_max_c=21.0,
            economic_target_c=19.6,
            preheat_block_id=block_id,
            preheat_block_budget_kwh=1.0 if block_id is not None else 0.0,
            preheat_block_cumulative_target_kwh=0.3 * (step + 1) if block_id is not None else 0.0,
            preheat_block_max_starts=1 if block_id is not None else 0,
            max_preheat_target_c=20.5,
            import_price_eur_kwh=0.22,
            export_price_eur_kwh=0.05,
        )
        for step in range(count)
    ]


def _build_flexibility(start_time: datetime, count: int, *, comfort_low_risk: bool = False, comfort_high_risk: bool = False) -> ThermalFlexibilityState:
    room_temp = 20.98 if comfort_high_risk else 19.2 if comfort_low_risk else 19.5
    no_heat_room = 18.95 if comfort_low_risk else 19.4
    return ThermalFlexibilityState(
        steps=[
            ThermalFlexibilityStep(
                index=step,
                timestamp_utc=start_time + timedelta(minutes=10 * step),
                temp_min_c=19.0,
                temp_max_c=21.0,
                economic_target_c=19.6,
                room_temp_c=room_temp,
                mass_temp_c=19.3,
                q_heat_eff_kw=0.5 if comfort_high_risk else 0.0,
                no_heat_room_temp_c=no_heat_room,
                no_heat_mass_temp_c=19.3,
                room_mass_delta_c=0.2,
                mass_deficit_to_economic_target_c=0.3,
                mass_deficit_to_preheat_target_c=0.7,
                normalized_storage_soc=0.25,
                comfort_headroom_c=max(21.0 - room_temp, 0.0),
                available_storage_kwh=1.0,
                expected_discharge_need_kwh=0.8,
                pv_surplus_forecast_kw=2.2 if not comfort_low_risk else 0.0,
                pv_surplus_window_kwh=0.8 if not comfort_low_risk else 0.0,
                post_solar_no_heat_drops_below_economic_target=not comfort_high_risk,
                post_solar_no_heat_drops_below_temp_min=comfort_low_risk,
            )
            for step in range(count)
        ]
    )


def _build_schedule(start_time: datetime, *, block_id: int | None) -> PreheatSchedule:
    if block_id is None:
        return PreheatSchedule(blocks=[], step_to_block_id=[None] * 3)
    return PreheatSchedule(
        blocks=[
            PreheatBlock(
                block_id=block_id,
                start_index=0,
                end_index=2,
                start_time_utc=start_time,
                end_time_utc=start_time + timedelta(minutes=20),
                available_surplus_kwh=1.0,
                available_storage_kwh=1.0,
                planned_charge_kwh=0.9,
                max_starts=1,
                min_run_steps=3,
                preferred_start_index=0,
                max_preheat_target_c=20.5,
                step_count=3,
                planned_run_steps=3,
            )
        ],
        step_to_block_id=[block_id, block_id, block_id],
        total_planned_charge_kwh=0.9,
    )


def _to_execution_target(step: MpcHorizonStep, target: ExecutionTargetStep) -> ExecutionTargetStep:
    return target.model_copy(update={"timestamp_utc": step.timestamp_utc})


def test_preheat_run_stays_committed_across_replans_until_min_runtime() -> None:
    sequencer = HeatPumpSequencer()
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    constraints = MpcConstraints(min_on_steps=0, min_off_steps=0)

    horizon = _build_horizon(start_time, 3, block_id=7)
    flexibility = _build_flexibility(start_time, 3)
    schedule = _build_schedule(start_time, block_id=7)

    targets, state = sequencer.build_execution_targets(
        horizon=horizon,
        flexibility_state=flexibility,
        schedule=schedule,
        constraints=constraints,
        sequencer_state=HeatPumpSequencerState(),
    )

    assert targets[0].hp_must_be_on is True
    assert targets[0].active_preheat_block_id == 7
    assert targets[0].active_run_id is not None

    advanced = sequencer.advance_state(
        state=HeatPumpSequencerState(),
        executed_step=horizon[0],
        executed_target=_to_execution_target(horizon[0], targets[0]),
        executed_hp_on=True,
        interval_minutes=10,
        preheat_charge_kwh=0.2,
    )

    next_targets, _ = sequencer.build_execution_targets(
        horizon=horizon[1:],
        flexibility_state=ThermalFlexibilityState(steps=flexibility.steps[1:]),
        schedule=schedule,
        constraints=constraints,
        sequencer_state=advanced,
    )

    assert next_targets[0].hp_must_be_on is True
    assert next_targets[0].active_run_id == targets[0].active_run_id


def test_max_starts_prevents_second_start_in_same_block() -> None:
    sequencer = HeatPumpSequencer()
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(start_time, 3, block_id=7)
    flexibility = _build_flexibility(start_time, 3)
    schedule = _build_schedule(start_time, block_id=7)

    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        flexibility_state=flexibility,
        schedule=schedule,
        constraints=MpcConstraints(),
        sequencer_state=HeatPumpSequencerState(starts_used_by_block={7: 1}),
    )

    assert targets[0].hp_start_allowed is False
    assert targets[0].starts_blocked_by_max_starts is True
    assert targets[0].hp_must_be_on is False


def test_lockout_blocks_restart() -> None:
    sequencer = HeatPumpSequencer()
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(start_time, 2, block_id=7)
    flexibility = _build_flexibility(start_time, 2)
    schedule = _build_schedule(start_time, block_id=7)

    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        flexibility_state=flexibility,
        schedule=schedule,
        constraints=MpcConstraints(),
        sequencer_state=HeatPumpSequencerState(
            mode="LOCKED_OUT",
            locked_off_until_utc=start_time + timedelta(minutes=15),
        ),
    )

    assert targets[0].hp_must_be_off is True
    assert targets[0].hp_start_allowed is False
    assert targets[0].starts_blocked_by_lockout is True


def test_comfort_low_risk_can_start_outside_preheat_block() -> None:
    sequencer = HeatPumpSequencer()
    start_time = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(start_time, 2, block_id=None)
    flexibility = _build_flexibility(start_time, 2, comfort_low_risk=True)
    schedule = _build_schedule(start_time, block_id=None)

    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        flexibility_state=flexibility,
        schedule=schedule,
        constraints=MpcConstraints(min_on_steps=2),
        sequencer_state=HeatPumpSequencerState(),
    )

    assert targets[0].hp_must_be_on is True
    assert targets[0].start_reason_hint == "comfort_low_risk"
    assert targets[0].active_preheat_block_id is None


def test_comfort_high_risk_can_stop_preheat_run() -> None:
    sequencer = HeatPumpSequencer()
    start_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    horizon = _build_horizon(start_time, 2, block_id=7)
    flexibility = _build_flexibility(start_time, 2, comfort_high_risk=True)
    schedule = _build_schedule(start_time, block_id=7)

    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        flexibility_state=flexibility,
        schedule=schedule,
        constraints=MpcConstraints(),
        sequencer_state=HeatPumpSequencerState(
            mode="PREHEAT_RUNNING",
            active_run_id="run-1",
            active_block_id=7,
            run_started_at_utc=start_time - timedelta(minutes=10),
            committed_on_until_utc=start_time + timedelta(minutes=5),
            previous_hp_on=True,
            on_steps=2,
            starts_used_by_block={7: 1},
        ),
    )

    assert targets[0].hp_must_be_off is True
    assert targets[0].stop_reason_hint == "comfort_high_risk"
