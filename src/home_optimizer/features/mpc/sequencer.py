from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
    HeatPumpSequencerSnapshot,
    HeatPumpSequencerState,
    MpcConstraints,
    MpcHorizonStep,
    PreheatBlock,
    PreheatSchedule,
    ThermalFlexibilityState,
)


class HeatPumpSequencerStateStore(Protocol):
    def load(self, key: str) -> HeatPumpSequencerState | None: ...

    def save(self, key: str, state: HeatPumpSequencerState) -> None: ...


class InMemoryHeatPumpSequencerStateStore:
    def __init__(self) -> None:
        self._states: dict[str, HeatPumpSequencerState] = {}

    def load(self, key: str) -> HeatPumpSequencerState | None:
        return self._states.get(key)

    def save(self, key: str, state: HeatPumpSequencerState) -> None:
        self._states[key] = state


@dataclass(slots=True)
class _ProjectedSequencerState:
    mode: str
    active_run_id: str | None
    active_block_id: int | None
    run_started_at_utc: datetime | None
    committed_on_until_utc: datetime | None
    locked_off_until_utc: datetime | None
    starts_used_by_block: dict[int, int]
    used_budget_by_block_kwh: dict[int, float]
    previous_hp_on: bool
    on_steps: int
    off_steps: int
    last_start_reason: str | None
    last_stop_reason: str | None


class HeatPumpSequencer:
    def __init__(self, *, state_store: HeatPumpSequencerStateStore | None = None) -> None:
        self.state_store = state_store or InMemoryHeatPumpSequencerStateStore()

    def load_state(self, key: str | None, fallback: HeatPumpSequencerState | None) -> HeatPumpSequencerState:
        if fallback is not None:
            return fallback
        if key:
            stored = self.state_store.load(key)
            if stored is not None:
                return stored
        return HeatPumpSequencerState()

    def save_state(self, key: str | None, state: HeatPumpSequencerState) -> None:
        if key:
            self.state_store.save(key, state)

    def build_execution_targets(
        self,
        *,
        horizon: list[MpcHorizonStep],
        flexibility_state: ThermalFlexibilityState,
        schedule: PreheatSchedule,
        constraints: MpcConstraints,
        sequencer_state: HeatPumpSequencerState,
    ) -> tuple[list[ExecutionTargetStep], HeatPumpSequencerState]:
        projected = _ProjectedSequencerState(
            mode=sequencer_state.mode,
            active_run_id=sequencer_state.active_run_id,
            active_block_id=sequencer_state.active_block_id,
            run_started_at_utc=sequencer_state.run_started_at_utc,
            committed_on_until_utc=sequencer_state.committed_on_until_utc,
            locked_off_until_utc=sequencer_state.locked_off_until_utc,
            starts_used_by_block=dict(sequencer_state.starts_used_by_block),
            used_budget_by_block_kwh=dict(sequencer_state.used_budget_by_block_kwh),
            previous_hp_on=sequencer_state.previous_hp_on,
            on_steps=sequencer_state.on_steps,
            off_steps=sequencer_state.off_steps,
            last_start_reason=sequencer_state.last_start_reason,
            last_stop_reason=sequencer_state.last_stop_reason,
        )
        blocks_by_id = {block.block_id: block for block in schedule.blocks}
        targets: list[ExecutionTargetStep] = []
        for index, step in enumerate(horizon):
            block = blocks_by_id.get(step.preheat_block_id) if step.preheat_block_id is not None else None
            flex_step = (
                flexibility_state.steps[index]
                if index < len(flexibility_state.steps)
                else None
            )
            snapshot = self._project_step(
                now=step.timestamp_utc,
                index=index,
                step=step,
                block=block,
                flex_step=flex_step,
                projected=projected,
                constraints=constraints,
                interval_minutes=self._infer_interval_minutes(horizon, index),
            )
            targets.append(
                ExecutionTargetStep(
                    timestamp_utc=step.timestamp_utc,
                    economic_target_c=float(
                        flex_step.economic_target_c
                        if flex_step is not None
                        else step.economic_target_c or step.temp_min_c
                    ),
                    preheat_target_c=(
                        float(block.max_preheat_target_c)
                        if block is not None
                        else float(step.economic_target_c or step.temp_min_c)
                    ),
                    active_preheat_block_id=snapshot.active_block_id,
                    remaining_block_budget_kwh=max(
                        self._remaining_block_budget(block, projected),
                        0.0,
                    ),
                    block_budget_share_kwh=float(step.preheat_budget_share_kwh),
                    block_cumulative_budget_target_kwh=float(step.preheat_block_cumulative_target_kwh),
                    storage_target_kwh=float(step.preheat_block_budget_kwh),
                    max_preheat_target_c=float(step.max_preheat_target_c or step.temp_max_c),
                    start_allowed_for_preheat=block is not None and snapshot.hp_start_allowed,
                    start_reason_hint=snapshot.start_reason,
                    sequencer_mode=snapshot.mode,
                    active_run_id=snapshot.active_run_id,
                    hp_must_be_on=snapshot.hp_must_be_on,
                    hp_must_be_off=snapshot.hp_must_be_off,
                    hp_start_allowed=snapshot.hp_start_allowed,
                    stop_reason_hint=snapshot.stop_reason,
                    committed_on_until_utc=snapshot.committed_on_until_utc,
                    locked_off_until_utc=snapshot.locked_off_until_utc,
                    starts_used_in_block=snapshot.starts_used_in_block,
                    run_budget_used_kwh=snapshot.run_budget_used_kwh,
                    starts_blocked_by_lockout=snapshot.starts_blocked_by_lockout,
                    starts_blocked_by_max_starts=snapshot.starts_blocked_by_max_starts,
                    starts_blocked_by_existing_commitment=snapshot.starts_blocked_by_existing_commitment,
                    stop_conditions=snapshot.stop_conditions,
                )
            )
        return targets, self._to_public_state(projected)

    def advance_state(
        self,
        *,
        state: HeatPumpSequencerState,
        executed_step: MpcHorizonStep,
        executed_target: ExecutionTargetStep,
        executed_hp_on: bool,
        interval_minutes: int,
        preheat_charge_kwh: float,
    ) -> HeatPumpSequencerState:
        next_data = state.model_dump()
        next_data["previous_hp_on"] = executed_hp_on
        if executed_hp_on:
            next_data["on_steps"] = int(state.on_steps) + 1
            next_data["off_steps"] = 0
        else:
            next_data["off_steps"] = int(state.off_steps) + 1
            next_data["on_steps"] = 0

        if executed_target.start_reason_hint and executed_hp_on and not state.previous_hp_on:
            next_data["last_start_reason"] = executed_target.start_reason_hint
            next_data["run_started_at_utc"] = executed_step.timestamp_utc
            next_data["active_run_id"] = executed_target.active_run_id
            next_data["active_block_id"] = executed_target.active_preheat_block_id
            next_data["committed_on_until_utc"] = executed_target.committed_on_until_utc
            if executed_target.active_preheat_block_id is not None:
                block_id = executed_target.active_preheat_block_id
                starts_used = dict(state.starts_used_by_block)
                starts_used[block_id] = starts_used.get(block_id, 0) + 1
                next_data["starts_used_by_block"] = starts_used
                next_data["mode"] = "PREHEAT_RUNNING"
            else:
                next_data["mode"] = "COMFORT_RUNNING"

        if executed_target.active_preheat_block_id is not None and preheat_charge_kwh > 0.0:
            block_id = executed_target.active_preheat_block_id
            used_budget = dict(next_data.get("used_budget_by_block_kwh", {}))
            used_budget[block_id] = used_budget.get(block_id, 0.0) + preheat_charge_kwh
            next_data["used_budget_by_block_kwh"] = used_budget

        if not executed_hp_on and state.previous_hp_on:
            next_data["last_stop_reason"] = executed_target.stop_reason_hint
            next_data["locked_off_until_utc"] = (
                executed_step.timestamp_utc + timedelta(minutes=interval_minutes * max(1, int(next_data["off_steps"])))
            )
            next_data["active_run_id"] = None
            next_data["active_block_id"] = None
            next_data["committed_on_until_utc"] = None
            next_data["mode"] = (
                "LOCKED_OUT" if executed_target.stop_reason_hint != "comfort_low_risk" else "IDLE"
            )

        if executed_target.locked_off_until_utc is not None:
            next_data["locked_off_until_utc"] = executed_target.locked_off_until_utc
        return HeatPumpSequencerState.model_validate(next_data)

    def _project_step(
        self,
        *,
        now: datetime,
        index: int,
        step: MpcHorizonStep,
        block: PreheatBlock | None,
        flex_step,
        projected: _ProjectedSequencerState,
        constraints: MpcConstraints,
        interval_minutes: int,
    ) -> HeatPumpSequencerSnapshot:
        comfort_low_risk = (
            flex_step is not None
            and (
                flex_step.no_heat_room_temp_c <= float(step.temp_min_c) + 0.05
                or flex_step.post_solar_no_heat_drops_below_temp_min
            )
        )
        current_below_comfort = (
            flex_step is not None
            and flex_step.room_temp_c < float(step.temp_min_c)
        )
        comfort_high_risk = (
            flex_step is not None
            and (
                projected.previous_hp_on
                and (
                    flex_step.room_temp_c >= float(step.temp_max_c) - 0.05
                    or flex_step.room_temp_c >= float(step.max_preheat_target_c or step.temp_max_c) - 0.05
                )
            )
        )
        starts_blocked_by_lockout = (
            projected.locked_off_until_utc is not None
            and now < projected.locked_off_until_utc
        )
        starts_blocked_by_existing_commitment = projected.mode in {"PREHEAT_RUNNING", "COMFORT_RUNNING"}
        starts_blocked_by_max_starts = (
            block is not None
            and projected.starts_used_by_block.get(block.block_id, 0) >= block.max_starts
        )

        hp_must_be_on = False
        hp_must_be_off = False
        hp_start_allowed = (
            not starts_blocked_by_lockout
            and not starts_blocked_by_existing_commitment
            and not starts_blocked_by_max_starts
        )
        start_reason: str | None = None
        stop_reason: str | None = None
        stop_conditions: list[str] = []

        if projected.mode == "LOCKED_OUT" and projected.locked_off_until_utc is not None and now >= projected.locked_off_until_utc:
            projected.mode = "IDLE"
            projected.locked_off_until_utc = None

        if projected.mode == "PREHEAT_RUNNING":
            remaining_budget_kwh = self._remaining_block_budget(block, projected)
            commitment_active = projected.committed_on_until_utc is not None and now < projected.committed_on_until_utc
            if comfort_high_risk:
                projected.mode = "SAFETY_STOP"
                stop_reason = "comfort_high_risk"
                stop_conditions.append("comfort_high_risk")
                hp_must_be_off = True
            elif commitment_active or (
                remaining_budget_kwh > 0.01
                and block is not None
                and now <= block.end_time_utc
                and not (flex_step and flex_step.post_solar_no_heat_drops_below_economic_target is False and flex_step.normalized_storage_soc >= 0.7)
            ):
                hp_must_be_on = True
            else:
                projected.mode = "LOCKED_OUT"
                projected.locked_off_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_off_steps, 1)
                )
                projected.active_run_id = None
                projected.active_block_id = None
                projected.committed_on_until_utc = None
                hp_must_be_off = True
                stop_reason = "budget_reached" if remaining_budget_kwh <= 0.01 else "post_solar_hold_ok"
                stop_conditions.append(stop_reason)
        elif projected.mode == "COMFORT_RUNNING":
            commitment_active = projected.committed_on_until_utc is not None and now < projected.committed_on_until_utc
            if comfort_high_risk:
                projected.mode = "SAFETY_STOP"
                hp_must_be_off = True
                stop_reason = "comfort_high_risk"
                stop_conditions.append("comfort_high_risk")
            elif commitment_active or current_below_comfort:
                hp_must_be_on = True
            else:
                projected.mode = "LOCKED_OUT"
                projected.locked_off_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_off_steps, 1)
                )
                projected.active_run_id = None
                projected.active_block_id = None
                projected.committed_on_until_utc = None
                hp_must_be_off = True
                stop_reason = "comfort_recovered"
                stop_conditions.append("comfort_recovered")
        elif projected.mode == "SAFETY_STOP":
            hp_must_be_off = True
            projected.locked_off_until_utc = now + timedelta(
                minutes=interval_minutes * max(constraints.min_off_steps, 1)
            )
            projected.mode = "LOCKED_OUT"
            stop_reason = "safety_stop"
            stop_conditions.append("safety_stop")
        else:
            if (
                block is not None
                and not starts_blocked_by_lockout
                and not starts_blocked_by_max_starts
                and self._remaining_block_budget(block, projected) > 0.01
                and not comfort_high_risk
            ):
                projected.mode = "PREHEAT_RUNNING"
                projected.active_block_id = block.block_id
                projected.active_run_id = f"{block.block_id}-{now.isoformat()}"
                projected.run_started_at_utc = now
                projected.committed_on_until_utc = now + timedelta(
                    minutes=interval_minutes * max(block.min_run_steps, constraints.min_on_steps, 1)
                )
                projected.starts_used_by_block[block.block_id] = (
                    projected.starts_used_by_block.get(block.block_id, 0) + 1
                )
                hp_must_be_on = True
                hp_start_allowed = True
                start_reason = "preheat_block"
            elif comfort_low_risk and not starts_blocked_by_lockout:
                projected.mode = "COMFORT_RUNNING"
                projected.active_block_id = None
                projected.active_run_id = f"comfort-{now.isoformat()}"
                projected.run_started_at_utc = now
                projected.committed_on_until_utc = now + timedelta(
                    minutes=interval_minutes * max(constraints.min_on_steps, 1)
                )
                hp_must_be_on = True
                hp_start_allowed = True
                start_reason = "comfort_low_risk"
            else:
                hp_must_be_off = starts_blocked_by_lockout
                hp_start_allowed = (
                    not starts_blocked_by_lockout
                    and not starts_blocked_by_max_starts
                )

        projected.previous_hp_on = hp_must_be_on
        if hp_must_be_on:
            projected.on_steps += 1
            projected.off_steps = 0
        elif hp_must_be_off:
            projected.off_steps += 1
            projected.on_steps = 0

        return HeatPumpSequencerSnapshot(
            mode=projected.mode,
            active_run_id=projected.active_run_id,
            active_block_id=projected.active_block_id,
            hp_must_be_on=hp_must_be_on,
            hp_must_be_off=hp_must_be_off,
            hp_start_allowed=hp_start_allowed,
            start_reason=start_reason or projected.last_start_reason,
            stop_reason=stop_reason or projected.last_stop_reason,
            committed_on_until_utc=projected.committed_on_until_utc,
            locked_off_until_utc=projected.locked_off_until_utc,
            starts_used_in_block=(
                projected.starts_used_by_block.get(projected.active_block_id, 0)
                if projected.active_block_id is not None
                else 0
            ),
            run_budget_used_kwh=(
                projected.used_budget_by_block_kwh.get(projected.active_block_id, 0.0)
                if projected.active_block_id is not None
                else 0.0
            ),
            starts_blocked_by_lockout=starts_blocked_by_lockout,
            starts_blocked_by_max_starts=starts_blocked_by_max_starts,
            starts_blocked_by_existing_commitment=starts_blocked_by_existing_commitment,
            run_target_budget_kwh=max(self._remaining_block_budget(block, projected), 0.0),
            stop_conditions=stop_conditions,
        )

    @staticmethod
    def _remaining_block_budget(block: PreheatBlock | None, projected: _ProjectedSequencerState) -> float:
        if block is None:
            return 0.0
        return max(
            float(block.planned_charge_kwh)
            - projected.used_budget_by_block_kwh.get(block.block_id, 0.0),
            0.0,
        )

    @staticmethod
    def _infer_interval_minutes(horizon: list[MpcHorizonStep], index: int) -> int:
        if len(horizon) <= 1:
            return 10
        if index < len(horizon) - 1:
            delta = horizon[index + 1].timestamp_utc - horizon[index].timestamp_utc
        else:
            delta = horizon[index].timestamp_utc - horizon[index - 1].timestamp_utc
        return max(int(delta.total_seconds() // 60), 1)

    @staticmethod
    def _to_public_state(projected: _ProjectedSequencerState) -> HeatPumpSequencerState:
        return HeatPumpSequencerState(
            mode=projected.mode,
            active_run_id=projected.active_run_id,
            active_block_id=projected.active_block_id,
            run_started_at_utc=projected.run_started_at_utc,
            committed_on_until_utc=projected.committed_on_until_utc,
            locked_off_until_utc=projected.locked_off_until_utc,
            starts_used_by_block=dict(projected.starts_used_by_block),
            used_budget_by_block_kwh=dict(projected.used_budget_by_block_kwh),
            previous_hp_on=projected.previous_hp_on,
            on_steps=projected.on_steps,
            off_steps=projected.off_steps,
            last_start_reason=projected.last_start_reason,
            last_stop_reason=projected.last_stop_reason,
        )
