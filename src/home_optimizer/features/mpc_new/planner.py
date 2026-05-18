from __future__ import annotations

from dataclasses import dataclass

from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    PreheatBlock,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    ThermalFlexibilityState,
)
from home_optimizer.features.mpc_new.models import (
    PreheatRunIntent,
    RejectedIntentCandidate,
    RunExecutionState,
    RunIntentPlan,
    RunIntentPlanningPolicy,
)


@dataclass(slots=True)
class _PlanningState:
    room_temp_c: float
    mass_temp_c: float | None
    q_heat_eff_kw: float
    hp_on: bool


@dataclass(slots=True)
class _IntentCandidate:
    intent: PreheatRunIntent
    score: float
    used_charge_kwh: float
    captured_pv_kwh: float
    imported_energy_kwh: float
    future_temp_min_violation_c: float
    future_economic_violation_c: float
    later_start_count: int
    comfort_high_risk_c: float


class RunSelectionPlanner:
    def build_plan(
        self,
        *,
        flexibility_state: ThermalFlexibilityState,
        constraints: MpcConstraints,
        interval_minutes: int,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: list[MpcHorizonStep],
        planning_policy: RunIntentPlanningPolicy | None = None,
        previous_plan: RunIntentPlan | None = None,
        execution_state: RunExecutionState | None = None,
    ) -> RunIntentPlan:
        resolved_policy = planning_policy or RunIntentPlanningPolicy()
        if not flexibility_state.steps or not horizon:
            return RunIntentPlan(diagnostics={"reason": "empty_horizon"})

        min_run_steps = (
            constraints.min_on_steps
            if constraints.min_on_steps > 0
            else max(1, round(30 / interval_minutes))
        )
        active_indices = [
            step.index
            for step in flexibility_state.steps
            if step.pv_surplus_forecast_kw > 0.0
            and step.expected_discharge_need_kwh > 0.0
        ]
        candidate_blocks = self._build_candidate_blocks(
            active_indices=active_indices,
            flexibility_state=flexibility_state,
            interval_minutes=interval_minutes,
            min_run_steps=min_run_steps,
        )
        planning_state = _PlanningState(
            room_temp_c=initial_state.room_temp_c,
            mass_temp_c=(
                initial_state.mass_temp_c
                if isinstance(initial_state, Rc2StateMpcInitialState)
                else None
            ),
            q_heat_eff_kw=initial_state.q_heat_eff_kw,
            hp_on=initial_state.hp_on,
        )
        candidates: list[_IntentCandidate] = []
        rejected: list[RejectedIntentCandidate] = []
        for block in candidate_blocks:
            candidate = self._best_intent_for_block(
                block=block,
                planning_state=planning_state,
                control_model=control_model,
                horizon=horizon,
                interval_minutes=interval_minutes,
                min_run_steps=min_run_steps,
            )
            if candidate is None:
                rejected.append(
                    RejectedIntentCandidate(
                        source_block_id=block.block_id,
                        reason="no_viable_run",
                        start_window_start_utc=block.start_time_utc,
                        start_window_end_utc=block.end_time_utc,
                        target_charge_kwh=block.planned_charge_kwh,
                    )
                )
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[: resolved_policy.max_selected_intents]
        rejected.extend(
            RejectedIntentCandidate(
                source_block_id=candidate.intent.source_block_id,
                reason="not_selected",
                score=candidate.score,
                start_window_start_utc=candidate.intent.start_window_start_utc,
                start_window_end_utc=candidate.intent.start_window_end_utc,
                target_charge_kwh=candidate.intent.target_charge_kwh,
                expected_captured_pv_kwh=candidate.intent.expected_captured_pv_kwh,
                expected_post_solar_hold_min_temp_c=(
                    candidate.intent.expected_post_solar_hold_min_temp_c
                ),
            )
            for candidate in candidates[resolved_policy.max_selected_intents :]
        )

        selected_intents = [candidate.intent for candidate in selected]
        selected_intents, keep_rejected = self._apply_replanning_hysteresis(
            selected_intents=selected_intents,
            rejected=rejected,
            previous_plan=previous_plan,
            execution_state=execution_state,
            policy=resolved_policy,
            horizon=horizon,
        )
        rejected = keep_rejected
        return RunIntentPlan(
            intents=selected_intents,
            rejected_candidates=rejected,
            selected_intent_count=len(selected_intents),
            total_target_charge_kwh=sum(
                intent.target_charge_kwh for intent in selected_intents
            ),
            diagnostics={
                "candidate_block_count": len(candidate_blocks),
                "selected_intent_count": len(selected_intents),
                "rejected_candidate_count": len(rejected),
            },
        )

    @staticmethod
    def _build_candidate_blocks(
        *,
        active_indices: list[int],
        flexibility_state: ThermalFlexibilityState,
        interval_minutes: int,
        min_run_steps: int,
    ) -> list[PreheatBlock]:
        if not active_indices:
            return []
        gap_tolerance_steps = max(1, round(30 / interval_minutes))
        blocks: list[PreheatBlock] = []
        current_indices: list[int] = []
        block_id = 0
        for index in active_indices:
            if current_indices and index > current_indices[-1] + 1 + gap_tolerance_steps:
                block = RunSelectionPlanner._create_block(
                    block_id=block_id,
                    step_indices=current_indices,
                    flexibility_state=flexibility_state,
                    interval_minutes=interval_minutes,
                    min_run_steps=min_run_steps,
                )
                if block is not None:
                    blocks.append(block)
                    block_id += 1
                current_indices = []
            current_indices.append(index)
        if current_indices:
            block = RunSelectionPlanner._create_block(
                block_id=block_id,
                step_indices=current_indices,
                flexibility_state=flexibility_state,
                interval_minutes=interval_minutes,
                min_run_steps=min_run_steps,
            )
            if block is not None:
                blocks.append(block)
        return blocks

    @staticmethod
    def _create_block(
        *,
        block_id: int,
        step_indices: list[int],
        flexibility_state: ThermalFlexibilityState,
        interval_minutes: int,
        min_run_steps: int,
    ) -> PreheatBlock | None:
        full_indices = list(range(step_indices[0], step_indices[-1] + 1))
        start_step = flexibility_state.steps[full_indices[0]]
        end_step = flexibility_state.steps[full_indices[-1]]
        dt_hours = interval_minutes / 60.0
        available_surplus_kwh = sum(
            flexibility_state.steps[index].pv_surplus_forecast_kw * dt_hours
            for index in full_indices
        )
        available_storage_kwh = max(
            (
                flexibility_state.steps[index].available_storage_kwh
                for index in full_indices
            ),
            default=0.0,
        )
        expected_discharge_need_kwh = max(
            (
                flexibility_state.steps[index].expected_discharge_need_kwh
                for index in full_indices
            ),
            default=0.0,
        )
        planned_charge_kwh = min(
            available_surplus_kwh,
            (
                expected_discharge_need_kwh
                if expected_discharge_need_kwh > 0.0
                else available_storage_kwh
            ),
        )
        max_hp_power_kw = max(
            (
                flexibility_state.steps[index].pv_surplus_forecast_kw
                for index in full_indices
            ),
            default=0.0,
        )
        minimum_useful_charge_kwh = max_hp_power_kw * min_run_steps * dt_hours * 0.5
        if planned_charge_kwh <= 0.0 or planned_charge_kwh < minimum_useful_charge_kwh:
            return None
        return PreheatBlock(
            block_id=block_id,
            start_index=full_indices[0],
            end_index=full_indices[-1],
            start_time_utc=start_step.timestamp_utc,
            end_time_utc=end_step.timestamp_utc,
            available_surplus_kwh=available_surplus_kwh,
            available_storage_kwh=available_storage_kwh,
            planned_charge_kwh=planned_charge_kwh,
            max_starts=1,
            min_run_steps=min_run_steps,
            max_preheat_target_c=max(
                flexibility_state.steps[index].temp_max_c for index in full_indices
            ),
            step_count=len(full_indices),
            reason="intent_candidate_from_sustained_surplus",
        )

    def _best_intent_for_block(
        self,
        *,
        block: PreheatBlock,
        planning_state: _PlanningState,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        horizon: list[MpcHorizonStep],
        interval_minutes: int,
        min_run_steps: int,
    ) -> _IntentCandidate | None:
        no_run_state_after_block = self._simulate_span(
            control_model=control_model,
            state=planning_state,
            horizon=horizon,
            start_index=block.start_index,
            end_index=block.end_index + 1,
            hp_on_indices=set(),
        )
        baseline_summary = self._simulate_future_no_heat_summary(
            control_model=control_model,
            state=no_run_state_after_block,
            horizon=horizon,
            start_index=block.end_index + 1,
        )
        baseline_storage_reference_c = (
            no_run_state_after_block.mass_temp_c
            if no_run_state_after_block.mass_temp_c is not None
            else no_run_state_after_block.room_temp_c
        )
        baseline_storage_state_c = max(
            baseline_storage_reference_c
            - float(
                horizon[block.end_index].economic_target_c
                or horizon[block.end_index].temp_min_c
            ),
            0.0,
        )
        best_candidate: _IntentCandidate | None = None
        latest_start_index = block.end_index - min_run_steps + 1
        if latest_start_index < block.start_index:
            return None

        for run_start_index in range(block.start_index, latest_start_index + 1):
            state_at_start = self._simulate_span(
                control_model=control_model,
                state=planning_state,
                horizon=horizon,
                start_index=block.start_index,
                end_index=run_start_index,
                hp_on_indices=set(),
            )
            max_run_steps = block.end_index - run_start_index + 1
            for planned_run_steps in range(min_run_steps, max_run_steps + 1):
                candidate = self._evaluate_candidate(
                    block=block,
                    run_start_index=run_start_index,
                    planned_run_steps=planned_run_steps,
                    state_at_start=state_at_start,
                    control_model=control_model,
                    horizon=horizon,
                    interval_minutes=interval_minutes,
                    baseline_summary=baseline_summary,
                    baseline_storage_state_c=baseline_storage_state_c,
                )
                if candidate is None:
                    continue
                if best_candidate is None or candidate.score > best_candidate.score:
                    best_candidate = candidate
        return best_candidate

    def _evaluate_candidate(
        self,
        *,
        block: PreheatBlock,
        run_start_index: int,
        planned_run_steps: int,
        state_at_start: _PlanningState,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        horizon: list[MpcHorizonStep],
        interval_minutes: int,
        baseline_summary: dict[str, float | bool | int],
        baseline_storage_state_c: float,
    ) -> _IntentCandidate | None:
        dt_hours = interval_minutes / 60.0
        run_end_index = min(run_start_index + planned_run_steps - 1, block.end_index)
        current_state = state_at_start
        used_charge_kwh = 0.0
        captured_pv_kwh = 0.0
        imported_energy_kwh = 0.0
        comfort_high_risk_c = 0.0
        for step_index in range(run_start_index, run_end_index + 1):
            step = horizon[step_index]
            hp_energy_kwh = float(step.hp_electric_power_forecast_kw) * dt_hours
            pv_capture_kwh = min(
                float(step.hp_electric_power_forecast_kw),
                max(
                    float(step.pv_available_power_forecast_kw)
                    - float(step.base_load_power_forecast_kw),
                    0.0,
                ),
            ) * dt_hours
            predicted_next_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=step,
                hp_on=True,
            )
            used_charge_kwh += hp_energy_kwh
            captured_pv_kwh += pv_capture_kwh
            imported_energy_kwh += max(hp_energy_kwh - pv_capture_kwh, 0.0)
            comfort_high_risk_c = max(
                comfort_high_risk_c,
                max(predicted_next_state.room_temp_c - float(step.temp_max_c), 0.0),
            )
            current_state = predicted_next_state
            if current_state.room_temp_c >= min(float(step.temp_max_c), block.max_preheat_target_c):
                break
            if used_charge_kwh >= block.planned_charge_kwh:
                break
        actual_run_steps = (run_end_index - run_start_index) + 1
        if actual_run_steps < block.min_run_steps:
            return None

        state_after_block = self._simulate_span(
            control_model=control_model,
            state=current_state,
            horizon=horizon,
            start_index=run_end_index + 1,
            end_index=block.end_index + 1,
            hp_on_indices=set(),
        )
        future_summary = self._simulate_future_no_heat_summary(
            control_model=control_model,
            state=state_after_block,
            horizon=horizon,
            start_index=block.end_index + 1,
        )
        storage_reference_c = (
            state_after_block.mass_temp_c
            if state_after_block.mass_temp_c is not None
            else state_after_block.room_temp_c
        )
        storage_state_gain_c = max(
            (
                storage_reference_c
                - float(
                    horizon[block.end_index].economic_target_c
                    or horizon[block.end_index].temp_min_c
                )
            )
            - baseline_storage_state_c,
            0.0,
        )
        score = self._score_candidate(
            captured_pv_kwh=captured_pv_kwh,
            imported_energy_kwh=imported_energy_kwh,
            storage_state_gain_c=storage_state_gain_c,
            baseline_min_room_temp_c=float(baseline_summary["min_room_temp_c"]),
            future_min_room_temp_c=float(future_summary["min_room_temp_c"]),
            baseline_end_room_temp_c=float(baseline_summary["end_room_temp_c"]),
            future_end_room_temp_c=float(future_summary["end_room_temp_c"]),
            baseline_temp_min_violation_c=float(baseline_summary["temp_min_violation_c"]),
            future_temp_min_violation_c=float(future_summary["temp_min_violation_c"]),
            baseline_economic_violation_c=float(baseline_summary["economic_violation_c"]),
            future_economic_violation_c=float(future_summary["economic_violation_c"]),
            baseline_later_start_count=int(baseline_summary["later_start_count"]),
            future_later_start_count=int(future_summary["later_start_count"]),
            comfort_high_risk_c=comfort_high_risk_c,
        )
        if score <= 0.0:
            return None
        planned_start = horizon[run_start_index].timestamp_utc
        planned_end = horizon[run_end_index].timestamp_utc
        intent = PreheatRunIntent(
            intent_id=f"intent-{block.block_id}-{planned_start.isoformat()}",
            run_type="preheat",
            source_block_id=block.block_id,
            start_window_start_utc=block.start_time_utc,
            start_window_end_utc=horizon[max(run_end_index, block.start_index)].timestamp_utc,
            latest_start_utc=horizon[block.end_index - block.min_run_steps + 1].timestamp_utc,
            planned_start_utc=planned_start,
            planned_end_utc=planned_end,
            min_run_steps=block.min_run_steps,
            target_charge_kwh=min(used_charge_kwh, block.planned_charge_kwh),
            target_post_solar_min_temp_c=float(future_summary["min_room_temp_c"]),
            max_preheat_target_c=block.max_preheat_target_c,
            max_starts=1,
            priority=100,
            valid_until_utc=block.end_time_utc,
            replacement_policy="replace_if_better",
            fallback_policy="comfort_low_only",
            score=score,
            expected_captured_pv_kwh=captured_pv_kwh,
            expected_post_solar_hold_min_temp_c=float(future_summary["min_room_temp_c"]),
            expected_used_charge_kwh=used_charge_kwh,
            expected_later_start_count=int(future_summary["later_start_count"]),
        )
        return _IntentCandidate(
            intent=intent,
            score=score,
            used_charge_kwh=used_charge_kwh,
            captured_pv_kwh=captured_pv_kwh,
            imported_energy_kwh=imported_energy_kwh,
            future_temp_min_violation_c=float(future_summary["temp_min_violation_c"]),
            future_economic_violation_c=float(future_summary["economic_violation_c"]),
            later_start_count=int(future_summary["later_start_count"]),
            comfort_high_risk_c=comfort_high_risk_c,
        )

    def _apply_replanning_hysteresis(
        self,
        *,
        selected_intents: list[PreheatRunIntent],
        rejected: list[RejectedIntentCandidate],
        previous_plan: RunIntentPlan | None,
        execution_state: RunExecutionState | None,
        policy: RunIntentPlanningPolicy,
        horizon: list[MpcHorizonStep],
    ) -> tuple[list[PreheatRunIntent], list[RejectedIntentCandidate]]:
        if (
            previous_plan is None
            or execution_state is None
            or execution_state.active_intent_id is None
        ):
            return selected_intents, rejected
        now = horizon[0].timestamp_utc
        active_intent = next(
            (
                intent
                for intent in previous_plan.intents
                if intent.intent_id == execution_state.active_intent_id
            ),
            None,
        )
        if active_intent is None or active_intent.valid_until_utc < now:
            return selected_intents, rejected
        committed = (
            execution_state.committed_on_until_utc is not None
            and now < execution_state.committed_on_until_utc
        )
        protected_until = execution_state.active_intent_started_at_utc
        if protected_until is not None:
            protected_until = protected_until.replace()
        if (
            execution_state.active_intent_started_at_utc is not None
            and (
                now - execution_state.active_intent_started_at_utc
            ).total_seconds()
            < (policy.minimum_time_before_replanning_active_intent_minutes * 60)
        ):
            kept_intent = active_intent.model_copy(
                update={"keep_reason": "minimum_time_before_replanning_active_intent"}
            )
            return [kept_intent], rejected
        if committed:
            kept_intent = active_intent.model_copy(update={"keep_reason": "committed_run"})
            return [kept_intent], rejected
        if not selected_intents:
            kept_intent = active_intent.model_copy(update={"keep_reason": "no_better_candidate"})
            return [kept_intent], rejected
        best_new = selected_intents[0]
        if (
            best_new.score
            <= active_intent.score
            + policy.keep_existing_intent_bonus
            + policy.minimum_improvement_to_replace_intent
        ):
            kept_intent = active_intent.model_copy(
                update={"keep_reason": "below_replacement_threshold"}
            )
            rejected.append(
                RejectedIntentCandidate(
                    source_block_id=best_new.source_block_id,
                    reason="replaced_by_active_intent_hysteresis",
                    score=best_new.score,
                    start_window_start_utc=best_new.start_window_start_utc,
                    start_window_end_utc=best_new.start_window_end_utc,
                    target_charge_kwh=best_new.target_charge_kwh,
                    expected_captured_pv_kwh=best_new.expected_captured_pv_kwh,
                    expected_post_solar_hold_min_temp_c=best_new.expected_post_solar_hold_min_temp_c,
                    keep_or_replace_reason="below_replacement_threshold",
                )
            )
            return [kept_intent], rejected
        replaced_active = best_new.model_copy(
            update={"replacement_reason": "improvement_above_threshold"}
        )
        return [replaced_active], rejected

    @staticmethod
    def _score_candidate(
        *,
        captured_pv_kwh: float,
        imported_energy_kwh: float,
        storage_state_gain_c: float,
        baseline_min_room_temp_c: float,
        future_min_room_temp_c: float,
        baseline_end_room_temp_c: float,
        future_end_room_temp_c: float,
        baseline_temp_min_violation_c: float,
        future_temp_min_violation_c: float,
        baseline_economic_violation_c: float,
        future_economic_violation_c: float,
        baseline_later_start_count: int,
        future_later_start_count: int,
        comfort_high_risk_c: float,
    ) -> float:
        temp_min_violation_reduction_c = max(
            baseline_temp_min_violation_c - future_temp_min_violation_c,
            0.0,
        )
        economic_violation_reduction_c = max(
            baseline_economic_violation_c - future_economic_violation_c,
            0.0,
        )
        later_start_reduction = max(
            baseline_later_start_count - future_later_start_count,
            0,
        )
        min_room_temp_gain_c = max(
            future_min_room_temp_c - baseline_min_room_temp_c,
            0.0,
        )
        end_room_temp_gain_c = max(
            future_end_room_temp_c - baseline_end_room_temp_c,
            0.0,
        )
        return (
            (12.0 * temp_min_violation_reduction_c)
            + (3.0 * economic_violation_reduction_c)
            + (2.0 * later_start_reduction)
            + (6.0 * captured_pv_kwh)
            + (2.5 * storage_state_gain_c)
            + (2.0 * min_room_temp_gain_c)
            + (1.0 * end_room_temp_gain_c)
            - (10.0 * comfort_high_risk_c)
            - (3.5 * imported_energy_kwh)
        )

    def _simulate_future_no_heat_summary(
        self,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        horizon: list[MpcHorizonStep],
        start_index: int,
    ) -> dict[str, float | bool | int]:
        current_state = state
        min_room_temp_c = current_state.room_temp_c
        end_room_temp_c = current_state.room_temp_c
        drops_below_economic = False
        drops_below_temp_min = False
        economic_violation_c = 0.0
        temp_min_violation_c = 0.0
        later_start_count = 0
        in_deficit = False
        for index in range(start_index, len(horizon)):
            step = horizon[index]
            current_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=step,
                hp_on=False,
            )
            min_room_temp_c = min(min_room_temp_c, current_state.room_temp_c)
            end_room_temp_c = current_state.room_temp_c
            economic_target_c = float(step.economic_target_c or step.temp_min_c)
            economic_gap_c = max(economic_target_c - current_state.room_temp_c, 0.0)
            temp_min_gap_c = max(float(step.temp_min_c) - current_state.room_temp_c, 0.0)
            economic_violation_c += economic_gap_c
            temp_min_violation_c += temp_min_gap_c
            if economic_gap_c > 0.0:
                drops_below_economic = True
                if not in_deficit:
                    later_start_count += 1
                    in_deficit = True
            else:
                in_deficit = False
            if temp_min_gap_c > 0.0:
                drops_below_temp_min = True
        return {
            "min_room_temp_c": min_room_temp_c,
            "end_room_temp_c": end_room_temp_c,
            "drops_below_economic": drops_below_economic,
            "drops_below_temp_min": drops_below_temp_min,
            "economic_violation_c": economic_violation_c,
            "temp_min_violation_c": temp_min_violation_c,
            "later_start_count": later_start_count,
        }

    def _simulate_span(
        self,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        horizon: list[MpcHorizonStep],
        start_index: int,
        end_index: int,
        hp_on_indices: set[int],
    ) -> _PlanningState:
        current_state = state
        for index in range(start_index, min(end_index, len(horizon))):
            current_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=horizon[index],
                hp_on=index in hp_on_indices,
            )
        return current_state

    @staticmethod
    def _simulate_step(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        step: MpcHorizonStep,
        hp_on: bool,
    ) -> _PlanningState:
        actuator_alpha = float(control_model.actuator_alpha)
        commanded_heat_kw = float(step.effective_heating_kw_forecast) if hp_on else 0.0
        q_heat_eff_kw = (
            actuator_alpha * state.q_heat_eff_kw
            + (1.0 - actuator_alpha) * commanded_heat_kw
        )
        if isinstance(control_model, Rc2StateThermalControlModel):
            next_room_temp_c, next_mass_temp_c = control_model.predict_next_state(
                room_temp_c=state.room_temp_c,
                mass_temp_c=(
                    state.mass_temp_c
                    if state.mass_temp_c is not None
                    else state.room_temp_c
                ),
                outdoor_temp_c=float(step.outdoor_temp_c),
                solar_gain_kw=float(step.solar_gain_kw),
                solar_gain_mass_kw=float(step.solar_gain_mass_kw or step.solar_gain_kw),
                heating_effect_kw=q_heat_eff_kw,
                occupied=float(step.occupied),
                hour_sin=float(step.hour_sin),
                hour_cos=float(step.hour_cos),
            )
            return _PlanningState(
                room_temp_c=next_room_temp_c,
                mass_temp_c=next_mass_temp_c,
                q_heat_eff_kw=max(q_heat_eff_kw, 0.0),
                hp_on=hp_on,
            )
        next_room_temp_c = control_model.predict_next_temperature(
            room_temp_c=state.room_temp_c,
            outdoor_temp_c=float(step.outdoor_temp_c),
            solar_gain_kw=float(step.solar_gain_kw),
            heating_effect_kw=q_heat_eff_kw,
            occupied=float(step.occupied),
        )
        return _PlanningState(
            room_temp_c=next_room_temp_c,
            mass_temp_c=None,
            q_heat_eff_kw=max(q_heat_eff_kw, 0.0),
            hp_on=hp_on,
        )
