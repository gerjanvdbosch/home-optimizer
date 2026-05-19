from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.features.mpc import (
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc_new import (
    IntentAwareMpcControllerRequest,
    IntentAwareMpcControllerService,
    IntentDrivenSequencer,
    PreheatRunIntent,
    RunExecutionState,
    RunIntentPlan,
)


def _build_horizon(
    *,
    start_time: datetime,
    steps: int,
    interval_minutes: int = 10,
    pv_by_step: dict[int, float] | None = None,
    outdoor_by_step: dict[int, float] | None = None,
    temp_min_c: float = 19.0,
    temp_max_c: float = 21.0,
) -> list[MpcHorizonStep]:
    pv_by_step = pv_by_step or {}
    outdoor_by_step = outdoor_by_step or {}
    return [
        MpcHorizonStep(
            timestamp_utc=start_time + timedelta(minutes=interval_minutes * step),
            outdoor_temp_c=outdoor_by_step.get(step, 6.0),
            solar_gain_kw=0.0,
            effective_heating_kw_forecast=2.0,
            hp_electric_power_forecast_kw=2.0,
            pv_available_power_forecast_kw=pv_by_step.get(step, 0.0),
            base_load_power_forecast_kw=0.3,
            occupied=0.0,
            target_temp_c=20.0,
            temp_min_c=temp_min_c,
            temp_max_c=temp_max_c,
            import_price_eur_kwh=0.30,
            export_price_eur_kwh=0.05,
        )
        for step in range(steps)
    ]


def _controller() -> IntentAwareMpcControllerService:
    return IntentAwareMpcControllerService()


def _model() -> Rc2StateThermalControlModel:
    return Rc2StateThermalControlModel(
        a11=0.94,
        a12=0.0,
        a21=0.0,
        a22=1.0,
        b_out_room=0.0,
        b_out_mass=0.0,
        b_solar_direct_room=0.0,
        b_heat_room=1.0,
        b_heat_mass=0.0,
        b_occ_room=0.0,
)


def test_single_pv_window_creates_one_intent_and_one_run() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=5,
        pv_by_step={0: 3.5, 1: 3.5, 2: 3.5, 3: 3.5},
        temp_max_c=22.0,
    )

    plan = _controller().plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.6, mass_temp_c=19.6, hp_on=False, off_steps=3),
    )

    assert plan.feasible is True
    assert plan.run_intent_plan is not None
    assert plan.run_intent_plan.selected_intent_count == 1
    assert sum(int(step.start) for step in plan.steps) == 1
    assert int(plan.diagnostics["comfort_fallback_run_count"]) == 0
    assert int(plan.diagnostics["starts_outside_intents"]) == 0


def test_multiple_candidate_blocks_select_only_best_intent() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=14,
        pv_by_step={1: 1.6, 2: 1.6, 8: 3.8, 9: 3.8, 10: 3.8},
        outdoor_by_step={11: -1.0, 12: -1.0, 13: -1.0},
        temp_max_c=22.0,
    )

    plan = _controller().plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=20.3, mass_temp_c=20.3, hp_on=False, off_steps=3),
    )

    assert plan.run_intent_plan is not None
    assert plan.run_intent_plan.selected_intent_count == 1
    assert plan.run_intent_plan.rejected_candidates
    rejected = next(
        candidate
        for candidate in plan.run_intent_plan.rejected_candidates
        if candidate.reason == "not_selected"
    )
    rejected_targets = [
        target
        for target in plan.execution_targets
        if rejected.start_window_start_utc is not None
        and rejected.start_window_end_utc is not None
        and rejected.start_window_start_utc <= target.timestamp_utc <= rejected.start_window_end_utc
    ]
    assert rejected_targets
    assert all(target.hp_start_allowed is False for target in rejected_targets)


def test_active_intent_stays_stable_for_small_forecast_changes() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=8,
        pv_by_step={0: 3.0, 1: 3.0, 2: 3.0},
        outdoor_by_step={3: 2.0, 4: 2.0, 5: 2.0},
    )
    controller = _controller()
    first_plan = controller.plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.2, mass_temp_c=19.2, hp_on=False, off_steps=3),
    )
    first_target = first_plan.execution_targets[0]
    next_state = controller.advance_execution_state(
        state=RunExecutionState(),
        executed_step=horizon[0],
        executed_target=first_target,
        executed_hp_on=True,
        interval_minutes=10,
        preheat_charge_kwh=0.33,
    )
    second_horizon = _build_horizon(
        start_time=start_time + timedelta(minutes=10),
        steps=8,
        pv_by_step={0: 2.8, 1: 3.0, 2: 3.0},
        outdoor_by_step={3: 2.0, 4: 2.2, 5: 2.0},
    )
    second_plan = controller.plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=second_horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
            previous_intent_plan=first_plan.run_intent_plan,
            run_execution_state=next_state,
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.6, mass_temp_c=19.6, hp_on=True, on_steps=1),
    )

    assert second_plan.run_intent_plan is not None
    assert first_plan.run_intent_plan is not None
    assert (
        second_plan.run_intent_plan.intents[0].intent_id
        == first_plan.run_intent_plan.intents[0].intent_id
    )
    assert second_plan.run_intent_plan.intents[0].keep_reason is not None


def test_intent_replaces_only_when_improvement_is_large() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    controller = _controller()
    previous_plan = RunIntentPlan(
        intents=[
            PreheatRunIntent(
                intent_id="legacy-intent",
                source_block_id=0,
                start_window_start_utc=start_time,
                start_window_end_utc=start_time + timedelta(minutes=20),
                latest_start_utc=start_time + timedelta(minutes=10),
                planned_start_utc=start_time,
                planned_end_utc=start_time + timedelta(minutes=20),
                min_run_steps=2,
                target_charge_kwh=0.4,
                target_post_solar_min_temp_c=19.1,
                max_preheat_target_c=21.0,
                max_starts=1,
                priority=100,
                valid_until_utc=start_time + timedelta(minutes=20),
                score=5.0,
            )
        ],
        selected_intent_count=1,
        total_target_charge_kwh=0.4,
    )
    previous_state = RunExecutionState(
        active_intent_id="legacy-intent",
        active_run_id="run-1",
        mode="IDLE",
        active_intent_started_at_utc=start_time - timedelta(minutes=60),
        previous_hp_on=False,
        off_steps=3,
    )
    second_horizon = _build_horizon(
        start_time=start_time,
        steps=10,
        pv_by_step={6: 4.2, 7: 4.2, 8: 4.2},
        outdoor_by_step={8: -2.0, 9: -2.0},
        temp_max_c=22.0,
    )
    second_plan = controller.plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=second_horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
            previous_intent_plan=previous_plan,
            run_execution_state=previous_state,
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.5, mass_temp_c=19.5, hp_on=False, off_steps=3),
    )

    assert second_plan.run_intent_plan is not None
    assert (
        second_plan.run_intent_plan.intents[0].replacement_reason
        == "improvement_above_threshold"
    )


def test_comfort_fallback_outside_intent_only_at_low_comfort_risk() -> None:
    start_time = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    no_pv_horizon = _build_horizon(
        start_time=start_time,
        steps=6,
        pv_by_step={},
        outdoor_by_step={0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 12.0},
    )
    controller = _controller()
    fallback_plan = controller.plan(
        IntentAwareMpcControllerRequest(interval_minutes=10, horizon=no_pv_horizon),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.02, mass_temp_c=19.02, hp_on=False, off_steps=4),
    )
    blocked_horizon = _build_horizon(
        start_time=start_time,
        steps=1,
        pv_by_step={},
        outdoor_by_step={0: 12.0},
    )
    blocked_plan = controller.plan(
        IntentAwareMpcControllerRequest(interval_minutes=10, horizon=blocked_horizon),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=21.2, mass_temp_c=21.2, hp_on=False, off_steps=4),
    )

    assert fallback_plan.run_intent_plan is not None
    assert fallback_plan.run_intent_plan.selected_intent_count == 0
    assert fallback_plan.execution_targets[0].hp_must_be_on is True
    assert fallback_plan.execution_targets[0].start_reason_hint == "emergency_comfort_low"
    assert blocked_plan.execution_targets[0].hp_start_allowed is False
    assert blocked_plan.execution_targets[0].starts_blocked_no_intent is True


def test_sequencer_preserves_active_intent_id_over_replans() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=8,
        pv_by_step={0: 3.2, 1: 3.2, 2: 3.2},
        outdoor_by_step={3: 2.0, 4: 2.0},
    )
    controller = _controller()
    first_plan = controller.plan(
        IntentAwareMpcControllerRequest(interval_minutes=10, horizon=horizon),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.1, mass_temp_c=19.1, hp_on=False, off_steps=4),
    )
    next_state = controller.advance_execution_state(
        state=RunExecutionState(),
        executed_step=horizon[0],
        executed_target=first_plan.execution_targets[0],
        executed_hp_on=True,
        interval_minutes=10,
        preheat_charge_kwh=0.33,
    )
    replan_horizon = _build_horizon(
        start_time=start_time + timedelta(minutes=10),
        steps=8,
        pv_by_step={0: 3.0, 1: 3.1, 2: 3.0},
        outdoor_by_step={3: 2.0, 4: 2.0},
    )
    second_plan = controller.plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=replan_horizon,
            previous_intent_plan=first_plan.run_intent_plan,
            run_execution_state=next_state,
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.4, mass_temp_c=19.4, hp_on=True, on_steps=1),
    )

    assert second_plan.execution_targets[0].active_intent_id == next_state.active_intent_id


def test_future_intent_metadata_does_not_leak_into_realized_state() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=6,
        pv_by_step={2: 3.0, 3: 3.0},
        outdoor_by_step={4: 2.0, 5: 2.0},
    )
    controller = _controller()
    plan = controller.plan(
        IntentAwareMpcControllerRequest(interval_minutes=10, horizon=horizon),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0, hp_on=False, off_steps=4),
    )
    first_target = plan.execution_targets[0]
    advanced = controller.advance_execution_state(
        state=RunExecutionState(),
        executed_step=horizon[0],
        executed_target=first_target,
        executed_hp_on=False,
        interval_minutes=10,
        preheat_charge_kwh=0.0,
    )

    assert advanced.active_intent_id is None
    assert advanced.active_run_id is None


def test_sequencer_only_starts_for_intent_or_safety() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    horizon = _build_horizon(
        start_time=start_time,
        steps=1,
        pv_by_step={},
        outdoor_by_step={0: 12.0},
    )
    sequencer = IntentDrivenSequencer()
    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        planning_state=_controller().planning_assessor.assess(
            interval_minutes=10,
            control_model=_model(),
            initial_state=Rc2StateMpcInitialState(room_temp_c=22.0, mass_temp_c=22.0, hp_on=False, off_steps=4),
            horizon=horizon,
            constraints=MpcConstraints(),
        ),
        intent_plan=RunIntentPlan(),
        constraints=MpcConstraints(),
        execution_state=RunExecutionState(),
        interval_minutes=10,
    )

    assert targets[0].hp_start_allowed is False


def test_comfort_bridge_start_uses_preheat_bridge_reason() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    intent = PreheatRunIntent(
        intent_id="bridge-intent",
        run_type="preheat",
        source_block_id=0,
        start_window_start_utc=start_time,
        start_window_end_utc=start_time + timedelta(minutes=20),
        latest_start_utc=start_time + timedelta(minutes=20),
        planned_start_utc=start_time + timedelta(minutes=10),
        planned_end_utc=start_time + timedelta(minutes=30),
        min_run_steps=2,
        target_charge_kwh=0.4,
        target_post_solar_min_temp_c=19.0,
        max_preheat_target_c=21.0,
        max_starts=1,
        priority=1,
        valid_until_utc=start_time + timedelta(minutes=30),
    )
    sequencer = IntentDrivenSequencer()
    horizon = _build_horizon(
        start_time=start_time,
        steps=1,
        pv_by_step={},
        outdoor_by_step={0: 12.0},
    )
    targets, _ = sequencer.build_execution_targets(
        horizon=horizon,
        planning_state=_controller().planning_assessor.assess(
            interval_minutes=10,
            control_model=_model(),
            initial_state=Rc2StateMpcInitialState(room_temp_c=19.02, mass_temp_c=19.02, hp_on=False, off_steps=4),
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        ),
        intent_plan=RunIntentPlan(intents=[intent], selected_intent_count=1),
        constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        execution_state=RunExecutionState(),
        interval_minutes=10,
    )

    assert targets[0].hp_must_be_on is True
    assert targets[0].start_reason_hint == "preheat_intent_comfort_bridge"


def test_missing_expired_intent_reset_gets_stop_reason() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sequencer = IntentDrivenSequencer()
    horizon = _build_horizon(
        start_time=start_time,
        steps=1,
        pv_by_step={},
        outdoor_by_step={0: 12.0},
    )
    targets, projected_state = sequencer.build_execution_targets(
        horizon=horizon,
        planning_state=_controller().planning_assessor.assess(
            interval_minutes=10,
            control_model=_model(),
            initial_state=Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0, hp_on=True, on_steps=2),
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        ),
        intent_plan=RunIntentPlan(),
        constraints=MpcConstraints(min_on_steps=2, min_off_steps=1),
        execution_state=RunExecutionState(
            active_intent_id="missing-intent",
            active_run_id="run-1",
            mode="PREHEAT_RUNNING",
            previous_hp_on=True,
            on_steps=2,
        ),
        interval_minutes=10,
    )

    assert targets[0].hp_must_be_off is True
    assert targets[0].stop_reason_hint == "missing_or_expired_intent_reset"
    assert projected_state.mode == "LOCKED_OUT"


def test_advance_execution_state_records_authority_ledger() -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    controller = _controller()
    horizon = _build_horizon(
        start_time=start_time,
        steps=4,
        pv_by_step={0: 3.5, 1: 3.5, 2: 0.0, 3: 0.0},
        temp_max_c=22.0,
    )
    plan = controller.plan(
        IntentAwareMpcControllerRequest(
            interval_minutes=10,
            horizon=horizon,
            constraints=MpcConstraints(min_on_steps=1, min_off_steps=1),
        ),
        control_model=_model(),
        initial_state=Rc2StateMpcInitialState(room_temp_c=19.4, mass_temp_c=19.4, hp_on=False, off_steps=4),
    )

    started = controller.advance_execution_state(
        state=RunExecutionState(),
        executed_step=horizon[0],
        executed_target=plan.execution_targets[0],
        executed_hp_on=True,
        interval_minutes=10,
        preheat_charge_kwh=0.2,
    )

    assert started.start_stop_ledger[-1].transition == "start"
    assert started.start_stop_ledger[-1].start_reason == "preheat_intent"
