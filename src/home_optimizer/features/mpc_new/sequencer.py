from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from home_optimizer.features.mpc.models import (
    MpcConstraints,
    MpcHorizonStep,
)
from home_optimizer.features.mpc_new.models import (
    AuthorityViolation,
    IntentPlanningState,
    PreheatRunIntent,
    RunExecutionState,
    RunIntentExecutionTargetStep,
    RunIntentPlan,
    StartStopLedgerEntry,
)


@dataclass(slots=True)
class _ProjectedState:
    active_intent_id: str | None
    active_run_id: str | None
    mode: str
    run_started_at_utc: datetime | None
    committed_on_until_utc: datetime | None
    locked_off_until_utc: datetime | None
    target_charge_kwh: float
    used_charge_kwh: float
    stop_reason: str | None
    start_reason: str | None
    previous_hp_on: bool
    on_steps: int
    off_steps: int
    active_source_block_id: int | None
    active_intent_started_at_utc: datetime | None


class IntentDrivenSequencer:
    def build_execution_targets(
        self,
        *,
        horizon: list[MpcHorizonStep],
        planning_state: IntentPlanningState,
        intent_plan: RunIntentPlan,
        constraints: MpcConstraints,
        execution_state: RunExecutionState | None,
        interval_minutes: int,
    ) -> tuple[list[RunIntentExecutionTargetStep], RunExecutionState]:
        projected = self._to_projected_state(execution_state or RunExecutionState())
        intents_by_id = {intent.intent_id: intent for intent in intent_plan.intents}
        targets: list[RunIntentExecutionTargetStep] = []
        for index, step in enumerate(horizon):
            flex_step = (
                planning_state.steps[index]
                if index < len(planning_state.steps)
                else None
            )
            eligible_intent = self._eligible_intent_for_step(
                intents=intent_plan.intents,
                now=step.timestamp_utc,
            )
            active_intent = (
                intents_by_id.get(projected.active_intent_id)
                if projected.active_intent_id
                else None
            )
            target, projected = self._project_step(
                now=step.timestamp_utc,
                step=step,
                flex_step=flex_step,
                eligible_intent=eligible_intent,
                active_intent=active_intent,
                state=projected,
                constraints=constraints,
                interval_minutes=interval_minutes,
            )
            targets.append(target)
        return targets, self._to_public_state(projected)

    def advance_state(
        self,
        *,
        state: RunExecutionState,
        executed_step: MpcHorizonStep,
        executed_target: RunIntentExecutionTargetStep,
        executed_hp_on: bool,
        interval_minutes: int,
        preheat_charge_kwh: float,
    ) -> RunExecutionState:
        next_data = state.model_dump()
        mode_before = state.mode
        next_data["previous_hp_on"] = executed_hp_on
        if executed_hp_on:
            next_data["on_steps"] = int(state.on_steps) + 1
            next_data["off_steps"] = 0
        else:
            next_data["off_steps"] = int(state.off_steps) + 1
            next_data["on_steps"] = 0

        actual_start = executed_hp_on and not state.previous_hp_on
        actual_stop = (not executed_hp_on) and state.previous_hp_on
        ledger = list(state.start_stop_ledger)
        violations = list(state.authority_violations)

        if actual_start:
            next_data["active_intent_id"] = executed_target.active_intent_id
            next_data["active_run_id"] = executed_target.active_run_id
            next_data["mode"] = executed_target.mode
            next_data["run_started_at_utc"] = executed_step.timestamp_utc
            next_data["active_intent_started_at_utc"] = executed_step.timestamp_utc
            next_data["committed_on_until_utc"] = executed_target.committed_on_until_utc
            next_data["target_charge_kwh"] = (
                preheat_charge_kwh + executed_target.target_charge_remaining_kwh
            )
            next_data["start_reason"] = executed_target.start_reason_hint
            next_data["stop_reason"] = None
            next_data["used_charge_kwh"] = preheat_charge_kwh
        elif executed_hp_on:
            next_data["mode"] = executed_target.mode
            next_data["active_intent_id"] = executed_target.active_intent_id
            next_data["active_run_id"] = executed_target.active_run_id
            next_data["committed_on_until_utc"] = executed_target.committed_on_until_utc
            next_data["used_charge_kwh"] = float(state.used_charge_kwh) + preheat_charge_kwh

        if actual_stop:
            next_data["stop_reason"] = executed_target.stop_reason_hint
            next_data["locked_off_until_utc"] = executed_step.timestamp_utc + timedelta(
                minutes=interval_minutes * max(1, state.off_steps + 1)
            )
            next_data["active_intent_id"] = None
            next_data["active_run_id"] = None
            next_data["committed_on_until_utc"] = None
            next_data["mode"] = "LOCKED_OUT"

        if executed_target.locked_off_until_utc is not None:
            next_data["locked_off_until_utc"] = executed_target.locked_off_until_utc
        next_state = RunExecutionState.model_validate(next_data)

        if actual_start:
            ledger.append(
                StartStopLedgerEntry(
                    timestamp=executed_step.timestamp_utc,
                    transition="start",
                    hp_on_previous=state.previous_hp_on,
                    hp_on_current=executed_hp_on,
                    start_reason=executed_target.start_reason_hint,
                    stop_reason=None,
                    intent_id=executed_target.active_intent_id,
                    intent_type=executed_target.intent_type,
                    active_run_id=executed_target.active_run_id,
                    sequencer_mode_before=mode_before,
                    sequencer_mode_after=next_state.mode,
                    hp_must_be_on=executed_target.hp_must_be_on,
                    hp_must_be_off=executed_target.hp_must_be_off,
                    hp_start_allowed=executed_target.hp_start_allowed,
                    comfort_low_risk=executed_target.comfort_low_risk,
                    comfort_high_risk=executed_target.comfort_high_risk,
                    predicted_min_temp_without_start=(
                        executed_target.predicted_min_temp_without_start
                    ),
                    room_temp_c=executed_target.room_temp_c,
                    mass_temp_c=executed_target.mass_temp_c,
                    q_heat_eff_kw=executed_target.q_heat_eff_kw,
                )
            )
        if actual_stop:
            ledger.append(
                StartStopLedgerEntry(
                    timestamp=executed_step.timestamp_utc,
                    transition="stop",
                    hp_on_previous=state.previous_hp_on,
                    hp_on_current=executed_hp_on,
                    start_reason=None,
                    stop_reason=executed_target.stop_reason_hint,
                    intent_id=state.active_intent_id,
                    intent_type=executed_target.intent_type,
                    active_run_id=state.active_run_id,
                    sequencer_mode_before=mode_before,
                    sequencer_mode_after=next_state.mode,
                    hp_must_be_on=executed_target.hp_must_be_on,
                    hp_must_be_off=executed_target.hp_must_be_off,
                    hp_start_allowed=executed_target.hp_start_allowed,
                    comfort_low_risk=executed_target.comfort_low_risk,
                    comfort_high_risk=executed_target.comfort_high_risk,
                    predicted_min_temp_without_start=(
                        executed_target.predicted_min_temp_without_start
                    ),
                    room_temp_c=executed_target.room_temp_c,
                    mass_temp_c=executed_target.mass_temp_c,
                    q_heat_eff_kw=executed_target.q_heat_eff_kw,
                )
            )

        if actual_start and executed_target.start_reason_hint is None:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="start_without_valid_reason",
                )
            )
        if actual_stop and executed_target.stop_reason_hint is None:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="stop_without_valid_reason",
                )
            )
        if actual_start and not executed_target.hp_start_allowed and not executed_target.hp_must_be_on:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="hp_start_allowed_false_but_start_true",
                )
            )
        if actual_start and executed_target.hp_must_be_on and executed_target.start_reason_hint is None:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="solver_created_start",
                )
            )
        if actual_start and executed_target.start_reason_hint == "emergency_comfort_low" and not executed_target.comfort_low_risk:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="comfort_fallback_without_comfort_low_risk",
                )
            )
        if actual_start and (
            executed_target.start_reason_hint
            not in {"preheat_intent", "comfort_recovery_intent", "preheat_intent_comfort_bridge", "emergency_comfort_low", "external_plant", "manual_override"}
        ):
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="start_without_intent_or_emergency",
                )
            )
        if executed_target.hp_must_be_on and not executed_hp_on:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="hp_must_be_on_true_but_hp_on_false",
                )
            )
        if executed_target.hp_must_be_off and executed_hp_on:
            violations.append(
                AuthorityViolation(
                    timestamp=executed_step.timestamp_utc,
                    violation="hp_must_be_off_true_but_hp_on_true",
                )
            )

        return next_state.model_copy(
            update={
                "start_stop_ledger": ledger,
                "authority_violations": violations,
            }
        )

    @staticmethod
    def _eligible_intent_for_step(
        *,
        intents: list[PreheatRunIntent],
        now,
    ) -> PreheatRunIntent | None:
        eligible = [
            intent
            for intent in intents
            if intent.start_window_start_utc <= now <= intent.start_window_end_utc
        ]
        if not eligible:
            return None
        return sorted(eligible, key=lambda intent: (-intent.priority, intent.planned_start_utc))[0]

    def _project_step(
        self,
        *,
        now,
        step: MpcHorizonStep,
        flex_step,
        eligible_intent: PreheatRunIntent | None,
        active_intent: PreheatRunIntent | None,
        state: _ProjectedState,
        constraints: MpcConstraints,
        interval_minutes: int,
    ) -> tuple[RunIntentExecutionTargetStep, RunExecutionState]:
        projected = _ProjectedState(**asdict(state))
        if (
            projected.mode in {"PREHEAT_RUNNING", "COMFORT_FALLBACK_RUNNING"}
            and (
                active_intent is None
                or now > active_intent.valid_until_utc
            )
            and projected.active_intent_id is not None
        ):
            projected.mode = "LOCKED_OUT"
            projected.active_intent_id = None
            projected.active_run_id = None
            projected.committed_on_until_utc = None
            projected.locked_off_until_utc = now + timedelta(
                minutes=interval_minutes * max(constraints.min_off_steps, 1)
            )
            target = RunIntentExecutionTargetStep(
                timestamp_utc=now,
                hp_must_be_on=False,
                hp_must_be_off=True,
                hp_start_allowed=False,
                stop_reason_hint="missing_or_expired_intent_reset",
                committed_on_until_utc=None,
                locked_off_until_utc=projected.locked_off_until_utc,
                mode=projected.mode,
                starts_blocked_no_intent=True,
            )
            projected.previous_hp_on = False
            projected.off_steps += 1
            projected.on_steps = 0
            return target, projected
        comfort_low_risk = bool(
            flex_step is not None
            and flex_step.post_solar_no_heat_drops_below_temp_min
        )
        comfort_high_risk = bool(
            flex_step is not None
            and projected.previous_hp_on
            and (
                flex_step.room_temp_c >= float(step.temp_max_c) - 0.05
                or (
                    active_intent is not None
                    and flex_step.room_temp_c >= active_intent.max_preheat_target_c - 0.05
                )
            )
        )
        predicted_min_temp_without_start = (
            min(
                flex_step.no_heat_room_temp_c,
                flex_step.post_solar_no_heat_min_temp_c
                if flex_step.post_solar_no_heat_min_temp_c is not None
                else flex_step.no_heat_room_temp_c,
            )
            if flex_step is not None
            else None
        )
        if (
            projected.mode == "LOCKED_OUT"
            and projected.locked_off_until_utc is not None
            and now >= projected.locked_off_until_utc
        ):
            projected.mode = "IDLE"
            projected.locked_off_until_utc = None

        starts_blocked_by_lockout = (
            projected.locked_off_until_utc is not None
            and now < projected.locked_off_until_utc
        )
        hp_must_be_on = False
        hp_must_be_off = False
        hp_start_allowed = False
        start_reason = None
        stop_reason = None
        active_intent_id = projected.active_intent_id
        active_run_id = projected.active_run_id
        mode = projected.mode

        if projected.mode == "PREHEAT_RUNNING" and active_intent is not None:
            commitment_active = (
                projected.committed_on_until_utc is not None
                and now < projected.committed_on_until_utc
            )
            if comfort_high_risk:
                mode = "SAFETY_STOP"
                hp_must_be_off = True
                stop_reason = "comfort_high_risk"
            elif commitment_active:
                hp_must_be_on = True
            elif now > active_intent.planned_end_utc or now > active_intent.valid_until_utc:
                mode = "LOCKED_OUT"
                hp_must_be_off = True
                stop_reason = "intent_completed"
                projected.locked_off_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_off_steps, 1)
                )
                active_intent_id = None
                active_run_id = None
            else:
                hp_must_be_on = True
        elif projected.mode == "COMFORT_FALLBACK_RUNNING":
            commitment_active = (
                projected.committed_on_until_utc is not None
                and now < projected.committed_on_until_utc
            )
            if comfort_high_risk:
                mode = "SAFETY_STOP"
                hp_must_be_off = True
                stop_reason = "comfort_high_risk"
            elif commitment_active or comfort_low_risk:
                hp_must_be_on = True
            else:
                mode = "LOCKED_OUT"
                hp_must_be_off = True
                stop_reason = "intent_completed"
                projected.locked_off_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_off_steps, 1)
                )
                active_intent_id = None
                active_run_id = None
        elif projected.mode == "SAFETY_STOP":
            hp_must_be_off = True
            mode = "LOCKED_OUT"
            stop_reason = "safety_stop"
            projected.locked_off_until_utc = now + timedelta(
                minutes=interval_minutes * max(constraints.min_off_steps, 1)
            )
            active_intent_id = None
            active_run_id = None
        else:
            if (
                eligible_intent is not None
                and not starts_blocked_by_lockout
                and now >= eligible_intent.planned_start_utc
                and now <= eligible_intent.latest_start_utc
            ):
                mode = "PREHEAT_RUNNING"
                active_intent_id = eligible_intent.intent_id
                active_run_id = f"{eligible_intent.intent_id}-{now.isoformat()}"
                projected.target_charge_kwh = eligible_intent.target_charge_kwh
                projected.used_charge_kwh = 0.0
                projected.committed_on_until_utc = now + timedelta(
                    minutes=interval_minutes
                    * max(
                        eligible_intent.min_run_steps,
                        constraints.min_on_steps,
                        1,
                    )
                )
                projected.active_source_block_id = eligible_intent.source_block_id
                hp_must_be_on = True
                hp_start_allowed = True
                start_reason = (
                    "comfort_recovery_intent"
                    if eligible_intent.run_type == "comfort_recovery"
                    else "preheat_intent"
                )
            elif comfort_low_risk and not starts_blocked_by_lockout:
                bridge_intent = eligible_intent
                if bridge_intent is not None:
                    mode = "PREHEAT_RUNNING"
                    active_intent_id = bridge_intent.intent_id
                    active_run_id = f"{bridge_intent.intent_id}-{now.isoformat()}"
                    projected.target_charge_kwh = bridge_intent.target_charge_kwh
                else:
                    mode = "COMFORT_FALLBACK_RUNNING"
                    active_intent_id = None
                    active_run_id = f"comfort-{now.isoformat()}"
                    projected.target_charge_kwh = 0.0
                projected.used_charge_kwh = 0.0
                projected.committed_on_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_on_steps, 1)
                )
                projected.active_source_block_id = None
                hp_must_be_on = True
                hp_start_allowed = True
                start_reason = (
                    "preheat_intent_comfort_bridge"
                    if bridge_intent is not None
                    else "emergency_comfort_low"
                )
            else:
                hp_must_be_off = starts_blocked_by_lockout
                hp_start_allowed = False
                mode = (
                    "PREHEAT_READY"
                    if eligible_intent is not None and not starts_blocked_by_lockout
                    else projected.mode
                )

        target = RunIntentExecutionTargetStep(
            timestamp_utc=now,
            active_intent_id=active_intent_id,
            active_run_id=active_run_id,
            eligible_intent_id=eligible_intent.intent_id if eligible_intent is not None else None,
            hp_must_be_on=hp_must_be_on,
            hp_must_be_off=hp_must_be_off,
            hp_start_allowed=hp_start_allowed,
            target_charge_remaining_kwh=max(
                projected.target_charge_kwh - projected.used_charge_kwh,
                0.0,
            ),
            max_preheat_target_c=(
                active_intent.max_preheat_target_c
                if active_intent is not None
                else eligible_intent.max_preheat_target_c
                if eligible_intent is not None
                else float(step.economic_target_c or step.temp_min_c)
            ),
            start_reason_hint=start_reason,
            stop_reason_hint=stop_reason,
            committed_on_until_utc=projected.committed_on_until_utc,
            locked_off_until_utc=projected.locked_off_until_utc,
            mode=mode,
            starts_blocked_no_intent=(eligible_intent is None and not comfort_low_risk),
            comfort_fallback_allowed=comfort_low_risk,
            intent_type=(
                active_intent.run_type
                if active_intent is not None
                else eligible_intent.run_type
                if eligible_intent is not None
                else None
            ),
            predicted_min_temp_without_start=predicted_min_temp_without_start,
            room_temp_c=flex_step.room_temp_c if flex_step is not None else None,
            mass_temp_c=flex_step.mass_temp_c if flex_step is not None else None,
            q_heat_eff_kw=flex_step.q_heat_eff_kw if flex_step is not None else 0.0,
            comfort_low_risk=comfort_low_risk,
            comfort_high_risk=comfort_high_risk,
        )
        projected.mode = mode
        projected.active_intent_id = active_intent_id
        projected.active_run_id = active_run_id
        projected.previous_hp_on = hp_must_be_on
        if hp_must_be_on:
            projected.on_steps += 1
            projected.off_steps = 0
        elif hp_must_be_off:
            projected.off_steps += 1
            projected.on_steps = 0
        return target, projected

    @staticmethod
    def _to_projected_state(state: RunExecutionState) -> _ProjectedState:
        state_data = state.model_dump(
            exclude={"start_stop_ledger", "authority_violations"}
        )
        return _ProjectedState(**state_data)

    @staticmethod
    def _to_public_state(state: _ProjectedState) -> RunExecutionState:
        return RunExecutionState.model_validate(asdict(state))
