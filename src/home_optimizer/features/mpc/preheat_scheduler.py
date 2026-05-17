from __future__ import annotations

from dataclasses import dataclass
import math

from home_optimizer.features.mpc.explain import rollout_without_heating
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    PreheatBlock,
    PreheatPlan,
    PreheatPlanStep,
    PreheatSchedule,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    ThermalFlexibilityState,
)


@dataclass(slots=True)
class _PlanningState:
    room_temp_c: float
    mass_temp_c: float | None
    q_heat_eff_kw: float
    hp_on: bool


class SpaceHeatingPreheatScheduler:
    def build_schedule(
        self,
        *,
        flexibility_state: ThermalFlexibilityState,
        constraints: MpcConstraints,
        interval_minutes: int,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel | None = None,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None = None,
        horizon: list[MpcHorizonStep] | None = None,
    ) -> PreheatSchedule:
        if not flexibility_state.steps:
            return PreheatSchedule(diagnostics={"reason": "empty_flexibility_state"})

        min_run_steps = (
            constraints.min_on_steps
            if constraints.min_on_steps > 0
            else max(1, round(30 / interval_minutes))
        )
        min_off_steps = (
            constraints.min_off_steps
            if constraints.min_off_steps > 0
            else max(1, round(30 / interval_minutes))
        )
        gap_tolerance_steps = max(1, round(30 / interval_minutes))
        active_indices = [
            step.index
            for step in flexibility_state.steps
            if step.pv_surplus_window_kwh > 0.0
            and step.available_storage_kwh > 0.0
            and step.expected_discharge_need_kwh > 0.0
        ]
        candidate_blocks = self._build_blocks_from_indices(
            active_indices=active_indices,
            flexibility_state=flexibility_state,
            useful_run_steps=min_run_steps,
            interval_minutes=interval_minutes,
            gap_tolerance_steps=gap_tolerance_steps,
        )
        required_preheat_charge_kwh = self._required_preheat_charge_kwh(
            flexibility_state=flexibility_state,
        )
        if control_model is None or initial_state is None or not horizon:
            selected_blocks = self._select_candidate_blocks_by_required_charge(
                candidate_blocks=candidate_blocks,
                required_preheat_charge_kwh=required_preheat_charge_kwh,
            )
            diagnostics = {
                "candidate_block_count": len(candidate_blocks),
                "selected_block_count": len(selected_blocks),
                "skipped_storage_sufficient_count": 0,
                "selection_mode": "greedy_fallback",
            }
        else:
            selected_blocks, diagnostics = self._plan_selected_blocks_sequentially(
                candidate_blocks=candidate_blocks,
                required_preheat_charge_kwh=required_preheat_charge_kwh,
                flexibility_state=flexibility_state,
                control_model=control_model,
                initial_state=initial_state,
                horizon=horizon,
                interval_minutes=interval_minutes,
                min_run_steps=min_run_steps,
                min_off_steps=min_off_steps,
            )
        step_to_block_id = [None for _ in flexibility_state.steps]
        for block in selected_blocks:
            for index in range(block.start_index, block.end_index + 1):
                step_to_block_id[index] = block.block_id
        return PreheatSchedule(
            blocks=selected_blocks,
            step_to_block_id=step_to_block_id,
            total_planned_charge_kwh=sum(block.planned_charge_kwh for block in selected_blocks),
            diagnostics={
                "required_preheat_charge_kwh": required_preheat_charge_kwh,
                "merged_gap_tolerance_steps": gap_tolerance_steps,
                **diagnostics,
            },
        )

    @staticmethod
    def _required_preheat_charge_kwh(
        *,
        flexibility_state: ThermalFlexibilityState,
    ) -> float:
        if not flexibility_state.steps:
            return 0.0
        max_available_storage_kwh = max(
            (step.available_storage_kwh for step in flexibility_state.steps),
            default=0.0,
        )
        peak_expected_discharge_need_kwh = max(
            (step.expected_discharge_need_kwh for step in flexibility_state.steps),
            default=0.0,
        )
        return max(0.0, min(max_available_storage_kwh, peak_expected_discharge_need_kwh))

    @staticmethod
    def _select_candidate_blocks_by_required_charge(
        *,
        candidate_blocks: list[PreheatBlock],
        required_preheat_charge_kwh: float,
    ) -> list[PreheatBlock]:
        if not candidate_blocks or required_preheat_charge_kwh <= 0.0:
            return []
        selected: list[PreheatBlock] = []
        remaining_need_kwh = required_preheat_charge_kwh
        ranked_candidates = sorted(
            enumerate(candidate_blocks),
            key=lambda item: (-item[1].planned_charge_kwh, item[1].start_index),
        )
        for candidate_index, candidate_block in ranked_candidates:
            planned_charge_kwh = min(candidate_block.planned_charge_kwh, remaining_need_kwh)
            if planned_charge_kwh <= 0.0:
                continue
            remaining_need_kwh = max(remaining_need_kwh - planned_charge_kwh, 0.0)
            selected.append(
                candidate_block.model_copy(
                    update={
                        "block_id": len(selected),
                        "candidate_block_id": candidate_index,
                        "selected": True,
                        "planned_charge_kwh": planned_charge_kwh,
                        "remaining_need_kwh": remaining_need_kwh,
                    }
                )
            )
            if remaining_need_kwh <= 0.0:
                break
        return selected

    def _plan_selected_blocks_sequentially(
        self,
        *,
        candidate_blocks: list[PreheatBlock],
        required_preheat_charge_kwh: float,
        flexibility_state: ThermalFlexibilityState,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: list[MpcHorizonStep],
        interval_minutes: int,
        min_run_steps: int,
        min_off_steps: int,
    ) -> tuple[list[PreheatBlock], dict[str, float | int]]:
        if not candidate_blocks or required_preheat_charge_kwh <= 0.0 or not horizon:
            return [], {
                "candidate_block_count": len(candidate_blocks),
                "selected_block_count": 0,
                "skipped_storage_sufficient_count": 0,
                "starts_in_preheat_blocks": 0,
                "starts_outside_preheat_blocks": 0,
                "late_comfort_starts_after_missed_preheat": 0,
            }

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
        selected_blocks: list[PreheatBlock] = []
        remaining_need_kwh = required_preheat_charge_kwh
        cursor_index = 0
        skipped_storage_sufficient_count = 0
        sorted_candidates = sorted(candidate_blocks, key=lambda block: block.start_index)
        for candidate_index, candidate_block in enumerate(sorted_candidates):
            planning_state = self._simulate_span(
                control_model=control_model,
                state=planning_state,
                horizon=horizon,
                start_index=cursor_index,
                end_index=candidate_block.start_index,
                hp_on_indices=set(),
            )
            no_heat_end_state = self._simulate_span(
                control_model=control_model,
                state=planning_state,
                horizon=horizon,
                start_index=candidate_block.start_index,
                end_index=candidate_block.end_index + 1,
                hp_on_indices=set(),
            )
            no_heat_hold = self._simulate_post_block_hold(
                control_model=control_model,
                state=no_heat_end_state,
                horizon=horizon,
                start_index=candidate_block.end_index + 1,
                interval_minutes=interval_minutes,
            )
            if remaining_need_kwh <= 0.0:
                cursor_index = candidate_block.end_index + 1
                planning_state = no_heat_end_state
                continue
            mass_temp_c = (
                planning_state.mass_temp_c
                if planning_state.mass_temp_c is not None
                else planning_state.room_temp_c
            )
            step_at_block_end = flexibility_state.steps[candidate_block.end_index]
            storage_sufficient = (
                not no_heat_hold["drops_below_economic"]
                and not no_heat_hold["drops_below_temp_min"]
                and mass_temp_c >= step_at_block_end.economic_target_c
                and step_at_block_end.normalized_storage_soc >= 0.6
            )
            if storage_sufficient:
                skipped_storage_sufficient_count += 1
                cursor_index = candidate_block.end_index + 1
                planning_state = no_heat_end_state
                continue

            planned_charge_kwh = min(
                candidate_block.planned_charge_kwh,
                remaining_need_kwh,
                max(
                    candidate_block.available_storage_kwh,
                    0.0,
                ),
            )
            if planned_charge_kwh <= 0.0:
                cursor_index = candidate_block.end_index + 1
                planning_state = no_heat_end_state
                continue

            run_summary = self._simulate_committed_run_in_block(
                control_model=control_model,
                state=planning_state,
                horizon=horizon,
                block=candidate_block,
                interval_minutes=interval_minutes,
                min_run_steps=min_run_steps,
                planned_charge_kwh=planned_charge_kwh,
            )
            used_charge_kwh = float(run_summary["used_charge_kwh"])
            remaining_need_kwh = max(
                remaining_need_kwh - used_charge_kwh,
                0.0,
            )
            selected_blocks.append(
                candidate_block.model_copy(
                    update={
                        "block_id": len(selected_blocks),
                        "candidate_block_id": candidate_index,
                        "selected": True,
                        "planned_charge_kwh": planned_charge_kwh,
                        "planned_run_steps": int(run_summary["planned_run_steps"]),
                        "preferred_start_index": int(run_summary["run_start_index"]),
                        "used_charge_kwh": used_charge_kwh,
                        "missed_charge_kwh": max(planned_charge_kwh - used_charge_kwh, 0.0),
                        "remaining_need_kwh": remaining_need_kwh,
                        "simulated_end_room_temp_c": float(run_summary["end_room_temp_c"]),
                        "simulated_end_mass_temp_c": run_summary["end_mass_temp_c"],
                        "post_solar_no_heat_min_temp_c": float(run_summary["hold_min_temp_c"]),
                        "post_solar_no_heat_end_temp_c": float(run_summary["hold_end_temp_c"]),
                        "post_solar_no_heat_drops_below_economic_target": bool(
                            run_summary["hold_drops_below_economic"]
                        ),
                        "post_solar_no_heat_drops_below_temp_min": bool(
                            run_summary["hold_drops_below_temp_min"]
                        ),
                        "starts_in_block": 1 if used_charge_kwh > 0.0 else 0,
                        "run_duration_minutes": float(run_summary["planned_run_steps"]) * interval_minutes,
                        "limit_reason": str(run_summary["stop_reason"]),
                    }
                )
            )
            planning_state = run_summary["end_state"]
            cursor_index = min(candidate_block.end_index + 1 + min_off_steps, len(horizon))

        return selected_blocks, {
            "candidate_block_count": len(sorted_candidates),
            "selected_block_count": len(selected_blocks),
            "skipped_storage_sufficient_count": skipped_storage_sufficient_count,
            "starts_in_preheat_blocks": sum(block.starts_in_block for block in selected_blocks),
            "starts_outside_preheat_blocks": 0,
            "late_comfort_starts_after_missed_preheat": 0,
        }

    def build(
        self,
        *,
        interval_minutes: int,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: list[MpcHorizonStep],
        constraints: MpcConstraints,
    ) -> PreheatPlan:
        if not horizon:
            return PreheatPlan(reason="empty_horizon")

        dt_hours = interval_minutes / 60.0
        max_import_price = max(
            (float(step.import_price_eur_kwh) for step in horizon),
            default=0.0,
        )
        window_steps = (
            constraints.pv_opportunity_window_steps
            if constraints.pv_opportunity_window_steps > 0
            else max(1, round(90 / interval_minutes))
        )
        useful_run_steps = max(
            constraints.min_on_steps,
            max(1, round(30 / interval_minutes)),
        )
        no_heat_rollout = rollout_without_heating(
            control_model=control_model,
            initial_state=initial_state,
            horizon=horizon,
        )
        pv_surplus_kwh = [
            max(
                0.0,
                float(step.pv_available_power_forecast_kw)
                - float(step.base_load_power_forecast_kw),
            )
            * dt_hours
            for step in horizon
        ]
        pv_surplus_window_kwh = [
            sum(pv_surplus_kwh[index : index + window_steps])
            for index in range(len(horizon))
        ]
        plan_steps: list[PreheatPlanStep] = []
        for index, step in enumerate(horizon):
            hp_electric_power_kw = max(float(step.hp_electric_power_forecast_kw), 1e-9)
            expected_min_run_electric_kwh = hp_electric_power_kw * useful_run_steps * dt_hours
            opportunity_score = min(
                pv_surplus_window_kwh[index] / expected_min_run_electric_kwh,
                1.0,
            )
            room_headroom_c = max(
                0.0,
                float(step.temp_max_c) - float(no_heat_rollout[index]),
            )
            room_band_c = max(float(step.temp_max_c) - float(step.temp_min_c), 1e-9)
            future_need_c = max(
                (
                    max(
                        0.0,
                        max(
                            float(future_step.temp_min_c),
                            float(future_step.target_temp_c or future_step.temp_min_c),
                        )
                        - float(no_heat_rollout[future_index]),
                    )
                    for future_index, future_step in enumerate(horizon[index:], start=index)
                ),
                default=0.0,
            )
            future_need_score = min(future_need_c / room_band_c, 1.0)
            economic_target_c = self._economic_target_c(
                step=step,
                future_need_score=future_need_score,
                max_import_price=max_import_price,
            )
            heat_gain_c = max(
                self._one_step_room_heat_gain_c(
                    control_model=control_model,
                    step=step,
                ),
                0.05,
            )
            storage_headroom_electric_kwh = (
                room_headroom_c / heat_gain_c
            ) * hp_electric_power_kw * dt_hours
            headroom_score = min(
                storage_headroom_electric_kwh / expected_min_run_electric_kwh,
                1.0,
            )
            preheat_score = opportunity_score * headroom_score * future_need_score
            preheat_active = preheat_score >= 0.2
            max_preheat_target_c = economic_target_c + (
                (float(step.temp_max_c) - economic_target_c) * preheat_score
            )
            preheat_budget_share_kwh = (
                min(
                    pv_surplus_kwh[index],
                    storage_headroom_electric_kwh,
                )
                if preheat_active
                else 0.0
            )
            plan_steps.append(
                PreheatPlanStep(
                    timestamp_utc=step.timestamp_utc,
                    economic_target_c=economic_target_c,
                    preheat_active=preheat_active,
                    preheat_opportunity_score=preheat_score,
                    max_preheat_target_c=min(
                        float(step.temp_max_c),
                        max(economic_target_c, max_preheat_target_c),
                    ),
                    preheat_budget_share_kwh=max(preheat_budget_share_kwh, 0.0),
                    pv_surplus_window_kwh=max(pv_surplus_window_kwh[index], 0.0),
                    storage_headroom_electric_kwh=max(storage_headroom_electric_kwh, 0.0),
                    reason=(
                        "pv_surplus_and_headroom"
                        if preheat_active
                        else "no_preheat_opportunity"
                    ),
                )
            )

        blocks = self._build_blocks(plan_steps)
        self._apply_block_metadata(
            plan_steps=plan_steps,
            blocks=blocks,
            pv_surplus_kwh=pv_surplus_kwh,
        )
        active_steps = [step for step in plan_steps if step.preheat_active]
        return PreheatPlan(
            steps=plan_steps,
            blocks=blocks,
            preheat_budget_electric_kwh=sum(
                step.preheat_budget_share_kwh for step in plan_steps
            ),
            preheat_window_start_utc=active_steps[0].timestamp_utc if active_steps else None,
            preheat_window_end_utc=active_steps[-1].timestamp_utc if active_steps else None,
            reason=(
                "pv_surplus_with_storage_headroom"
                if active_steps
                else "no_pv_surplus_with_headroom"
            ),
        )

    @staticmethod
    def _build_blocks_from_indices(
        *,
        active_indices: list[int],
        flexibility_state: ThermalFlexibilityState,
        useful_run_steps: int,
        interval_minutes: int,
        gap_tolerance_steps: int,
    ) -> list[PreheatBlock]:
        if not active_indices:
            return []
        blocks: list[PreheatBlock] = []
        current_block_indices: list[int] = []
        block_id = 0
        for index in active_indices:
            if current_block_indices and index > current_block_indices[-1] + 1 + gap_tolerance_steps:
                block = SpaceHeatingPreheatScheduler._create_schedule_block(
                    block_id=block_id,
                    step_indices=current_block_indices,
                    flexibility_state=flexibility_state,
                    useful_run_steps=useful_run_steps,
                    interval_minutes=interval_minutes,
                )
                if block is not None:
                    blocks.append(block)
                    block_id += 1
                current_block_indices = []
            current_block_indices.append(index)
        if current_block_indices:
            block = SpaceHeatingPreheatScheduler._create_schedule_block(
                block_id=block_id,
                step_indices=current_block_indices,
                flexibility_state=flexibility_state,
                useful_run_steps=useful_run_steps,
                interval_minutes=interval_minutes,
            )
            if block is not None:
                blocks.append(block)
        return blocks

    @staticmethod
    def _create_schedule_block(
        *,
        block_id: int,
        step_indices: list[int],
        flexibility_state: ThermalFlexibilityState,
        useful_run_steps: int,
        interval_minutes: int,
    ) -> PreheatBlock | None:
        full_range_indices = list(range(step_indices[0], step_indices[-1] + 1))
        start_step = flexibility_state.steps[full_range_indices[0]]
        end_step = flexibility_state.steps[full_range_indices[-1]]
        dt_hours = interval_minutes / 60.0
        available_surplus_kwh = sum(
            flexibility_state.steps[index].pv_surplus_forecast_kw * dt_hours
            for index in full_range_indices
        )
        available_storage_kwh = max(
            (flexibility_state.steps[index].available_storage_kwh for index in full_range_indices),
            default=0.0,
        )
        expected_discharge_need_kwh = max(
            (
                flexibility_state.steps[index].expected_discharge_need_kwh
                for index in full_range_indices
            ),
            default=0.0,
        )
        cumulative_future_need_kwh = sum(
            flexibility_state.steps[index].expected_discharge_need_kwh
            for index in full_range_indices
        )
        useful_discharge_need_kwh = max(
            expected_discharge_need_kwh,
            min(cumulative_future_need_kwh * 0.25, available_storage_kwh * 0.75),
        )
        planned_charge_kwh = min(
            available_surplus_kwh,
            available_storage_kwh,
            useful_discharge_need_kwh if useful_discharge_need_kwh > 0.0 else available_storage_kwh,
        )
        max_hp_power_kw = max(
            (
                flexibility_state.steps[index].pv_surplus_forecast_kw
                for index in full_range_indices
            ),
            default=0.0,
        )
        minimum_useful_charge_kwh = max_hp_power_kw * useful_run_steps * dt_hours * 0.5
        if planned_charge_kwh < minimum_useful_charge_kwh:
            return None
        preferred_start_index = max(
            full_range_indices,
            key=lambda index: flexibility_state.steps[index].pv_surplus_forecast_kw,
        )
        max_preheat_target_c = max(
            flexibility_state.steps[index].temp_min_c
            + min(
                flexibility_state.steps[index].comfort_headroom_c,
                (
                    flexibility_state.steps[index].available_storage_kwh
                    / max(available_storage_kwh, 1e-9)
                )
                * flexibility_state.steps[index].comfort_headroom_c,
            )
            for index in full_range_indices
        )
        return PreheatBlock(
            block_id=block_id,
            start_index=full_range_indices[0],
            end_index=full_range_indices[-1],
            start_time_utc=start_step.timestamp_utc,
            end_time_utc=end_step.timestamp_utc,
            available_surplus_kwh=max(available_surplus_kwh, 0.0),
            available_storage_kwh=max(available_storage_kwh, 0.0),
            planned_charge_kwh=max(planned_charge_kwh, 0.0),
            max_starts=1,
            min_run_steps=useful_run_steps,
            preferred_start_index=preferred_start_index,
            max_preheat_target_c=max_preheat_target_c,
            step_count=len(full_range_indices),
            reason="sustained_pv_surplus_with_storage",
        )

    @staticmethod
    def _build_blocks(plan_steps: list[PreheatPlanStep]) -> list[PreheatBlock]:
        blocks: list[PreheatBlock] = []
        active_indices = [index for index, step in enumerate(plan_steps) if step.preheat_active]
        if not active_indices:
            return blocks

        current_block_indices: list[int] = []
        block_id = 0
        gap_tolerance_steps = 1
        for index in active_indices:
            if (
                current_block_indices
                and index > current_block_indices[-1] + 1 + gap_tolerance_steps
            ):
                blocks.append(
                    SpaceHeatingPreheatScheduler._create_block(
                        block_id=block_id,
                        step_indices=current_block_indices,
                        plan_steps=plan_steps,
                    )
                )
                block_id += 1
                current_block_indices = []
            current_block_indices.append(index)
        if current_block_indices:
            blocks.append(
                SpaceHeatingPreheatScheduler._create_block(
                    block_id=block_id,
                    step_indices=current_block_indices,
                    plan_steps=plan_steps,
                )
            )
        return blocks

    @staticmethod
    def _create_block(
        *,
        block_id: int,
        step_indices: list[int],
        plan_steps: list[PreheatPlanStep],
    ) -> PreheatBlock:
        start_step = plan_steps[step_indices[0]]
        end_step = plan_steps[step_indices[-1]]
        full_range_indices = list(range(step_indices[0], step_indices[-1] + 1))
        available_surplus_kwh = sum(
            plan_steps[index].preheat_budget_share_kwh for index in full_range_indices
        )
        available_storage_kwh = max(
            (plan_steps[index].storage_headroom_electric_kwh for index in full_range_indices),
            default=0.0,
        )
        planned_charge_kwh = min(available_surplus_kwh, available_storage_kwh)
        return PreheatBlock(
            block_id=block_id,
            start_index=full_range_indices[0],
            end_index=full_range_indices[-1],
            start_time_utc=start_step.timestamp_utc,
            end_time_utc=end_step.timestamp_utc,
            available_surplus_kwh=max(available_surplus_kwh, 0.0),
            available_storage_kwh=max(available_storage_kwh, 0.0),
            planned_charge_kwh=max(planned_charge_kwh, 0.0),
            max_starts=1,
            step_count=len(full_range_indices),
            reason="clustered_pv_surplus_window",
        )

    @staticmethod
    def _apply_block_metadata(
        *,
        plan_steps: list[PreheatPlanStep],
        blocks: list[PreheatBlock],
        pv_surplus_kwh: list[float],
    ) -> None:
        if not blocks:
            return
        for block in blocks:
            block_step_indices = [
                index
                for index, step in enumerate(plan_steps)
                if step.timestamp_utc >= block.start_time_utc
                and step.timestamp_utc <= block.end_time_utc
            ]
            if not block_step_indices:
                continue
            total_surplus_kwh = sum(max(pv_surplus_kwh[index], 0.0) for index in block_step_indices)
            opportunity_score = max(
                plan_steps[index].preheat_opportunity_score for index in block_step_indices
            )
            max_target_c = max(
                plan_steps[index].max_preheat_target_c for index in block_step_indices
            )
            for index in block_step_indices:
                step = plan_steps[index]
                surplus_share = (
                    max(pv_surplus_kwh[index], 0.0) / total_surplus_kwh
                    if total_surplus_kwh > 0.0
                    else 1.0 / len(block_step_indices)
                )
                assigned_budget_kwh = block.planned_charge_kwh * surplus_share
                plan_steps[index] = step.model_copy(
                    update={
                        "preheat_active": True,
                        "preheat_block_id": block.block_id,
                        "preheat_block_budget_kwh": block.planned_charge_kwh,
                        "preheat_block_max_starts": block.max_starts,
                        "preheat_budget_share_kwh": assigned_budget_kwh,
                        "preheat_opportunity_score": opportunity_score,
                        "max_preheat_target_c": max_target_c,
                        "reason": (
                            "preheat_block_bridge_gap"
                            if not step.preheat_active
                            else "preheat_block_active"
                        ),
                    }
                )

    def _simulate_span(
        self,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        horizon: list[MpcHorizonStep],
        start_index: int,
        end_index: int,
        hp_on_indices: set[int],
        solar_scale: float = 1.0,
    ) -> _PlanningState:
        current_state = state
        for index in range(start_index, min(end_index, len(horizon))):
            current_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=horizon[index],
                hp_on=index in hp_on_indices,
                solar_scale=solar_scale,
            )
        return current_state

    def _simulate_post_block_hold(
        self,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        horizon: list[MpcHorizonStep],
        start_index: int,
        interval_minutes: int,
    ) -> dict[str, float | bool]:
        hold_steps = max(1, round(120 / interval_minutes))
        current_state = state
        min_room_temp_c = current_state.room_temp_c
        end_room_temp_c = current_state.room_temp_c
        drops_below_economic = False
        drops_below_temp_min = False
        for index in range(start_index, min(start_index + hold_steps, len(horizon))):
            step = horizon[index]
            current_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=step,
                hp_on=False,
                solar_scale=0.25,
            )
            min_room_temp_c = min(min_room_temp_c, current_state.room_temp_c)
            end_room_temp_c = current_state.room_temp_c
            if current_state.room_temp_c < float(step.economic_target_c or step.temp_min_c):
                drops_below_economic = True
            if current_state.room_temp_c < float(step.temp_min_c):
                drops_below_temp_min = True
        return {
            "min_room_temp_c": min_room_temp_c,
            "end_room_temp_c": end_room_temp_c,
            "drops_below_economic": drops_below_economic,
            "drops_below_temp_min": drops_below_temp_min,
        }

    def _simulate_committed_run_in_block(
        self,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        horizon: list[MpcHorizonStep],
        block: PreheatBlock,
        interval_minutes: int,
        min_run_steps: int,
        planned_charge_kwh: float,
    ) -> dict[str, object]:
        dt_hours = interval_minutes / 60.0
        run_start_index = block.preferred_start_index or block.start_index
        run_start_index = min(max(run_start_index, block.start_index), block.end_index)
        current_state = self._simulate_span(
            control_model=control_model,
            state=state,
            horizon=horizon,
            start_index=block.start_index,
            end_index=run_start_index,
            hp_on_indices=set(),
        )
        used_charge_kwh = 0.0
        planned_run_steps = 0
        stop_reason = "block_ended"
        step_index = run_start_index
        while step_index <= block.end_index:
            step = horizon[step_index]
            predicted_next_state = self._simulate_step(
                control_model=control_model,
                state=current_state,
                step=step,
                hp_on=True,
            )
            used_charge_kwh += float(step.hp_electric_power_forecast_kw) * dt_hours
            planned_run_steps += 1
            current_state = predicted_next_state
            if current_state.room_temp_c >= min(float(step.temp_max_c), block.max_preheat_target_c):
                stop_reason = "comfort_high_risk"
                if planned_run_steps >= min_run_steps:
                    break
            if used_charge_kwh >= planned_charge_kwh and planned_run_steps >= min_run_steps:
                stop_reason = "budget_reached"
                break
            step_index += 1

        hold_summary = self._simulate_post_block_hold(
            control_model=control_model,
            state=current_state,
            horizon=horizon,
            start_index=min(block.end_index + 1, len(horizon)),
            interval_minutes=interval_minutes,
        )
        future_need_kwh = 0.0
        for future_index in range(min(block.end_index + 1, len(horizon)), len(horizon)):
            future_step = horizon[future_index]
            deficit_c = max(
                float(future_step.economic_target_c or future_step.temp_min_c) - hold_summary["end_room_temp_c"],
                0.0,
            )
            if deficit_c <= 0.0:
                continue
            heat_gain_c = max(
                self._one_step_room_heat_gain_c(
                    control_model=control_model,
                    step=future_step,
                ),
                0.05,
            )
            future_need_kwh += (
                deficit_c / heat_gain_c
            ) * max(float(future_step.hp_electric_power_forecast_kw), 1e-9) * dt_hours

        return {
            "run_start_index": run_start_index,
            "planned_run_steps": planned_run_steps,
            "used_charge_kwh": used_charge_kwh,
            "stop_reason": stop_reason,
            "end_state": current_state,
            "end_room_temp_c": current_state.room_temp_c,
            "end_mass_temp_c": current_state.mass_temp_c,
            "hold_min_temp_c": hold_summary["min_room_temp_c"],
            "hold_end_temp_c": hold_summary["end_room_temp_c"],
            "hold_drops_below_economic": hold_summary["drops_below_economic"],
            "hold_drops_below_temp_min": hold_summary["drops_below_temp_min"],
            "remaining_need_kwh": future_need_kwh,
        }

    @staticmethod
    def _simulate_step(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        state: _PlanningState,
        step: MpcHorizonStep,
        hp_on: bool,
        solar_scale: float = 1.0,
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
                mass_temp_c=state.mass_temp_c if state.mass_temp_c is not None else state.room_temp_c,
                outdoor_temp_c=float(step.outdoor_temp_c),
                solar_gain_kw=float(step.solar_gain_kw) * solar_scale,
                solar_gain_mass_kw=float(step.solar_gain_mass_kw or step.solar_gain_kw) * solar_scale,
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
            solar_gain_kw=float(step.solar_gain_kw) * solar_scale,
            heating_effect_kw=q_heat_eff_kw,
            occupied=float(step.occupied),
        )
        return _PlanningState(
            room_temp_c=next_room_temp_c,
            mass_temp_c=None,
            q_heat_eff_kw=max(q_heat_eff_kw, 0.0),
            hp_on=hp_on,
        )

    @staticmethod
    def _economic_target_c(
        *,
        step: MpcHorizonStep,
        future_need_score: float,
        max_import_price: float,
    ) -> float:
        comfort_floor_c = float(step.temp_min_c)
        base_target_c = float(step.target_temp_c or step.temp_min_c)
        base_tracking_band_c = max(base_target_c - comfort_floor_c, 0.0)
        if base_tracking_band_c <= 0.0:
            return comfort_floor_c
        price_norm = (
            min(float(step.import_price_eur_kwh) / max_import_price, 1.0)
            if max_import_price > 0.0
            else 0.0
        )
        economic_score = max(
            0.0,
            min(
                1.0,
                (0.2 + (0.8 * future_need_score))
                * (1.0 - (0.75 * price_norm)),
            ),
        )
        return comfort_floor_c + (base_tracking_band_c * economic_score)

    @staticmethod
    def _one_step_room_heat_gain_c(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        step: MpcHorizonStep,
    ) -> float:
        if isinstance(control_model, Rc2StateThermalControlModel):
            room_gain = max(control_model.b_heat_room, 0.0)
            mass_leak_gain = max(control_model.b_heat_mass, 0.0) * 0.5
            return (room_gain + mass_leak_gain) * float(step.effective_heating_kw_forecast)
        return max(control_model.b_heat, 0.0) * float(step.effective_heating_kw_forecast)
