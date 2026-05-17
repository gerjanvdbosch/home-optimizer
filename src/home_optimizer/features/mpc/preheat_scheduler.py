from __future__ import annotations

import math

from home_optimizer.features.mpc.explain import rollout_without_heating
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    PreheatSchedule,
    PreheatBlock,
    PreheatPlan,
    PreheatPlanStep,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    ThermalFlexibilityState,
)


class SpaceHeatingPreheatScheduler:
    def build_schedule(
        self,
        *,
        flexibility_state: ThermalFlexibilityState,
        constraints: MpcConstraints,
        interval_minutes: int,
    ) -> PreheatSchedule:
        if not flexibility_state.steps:
            return PreheatSchedule(diagnostics={"reason": "empty_flexibility_state"})

        useful_run_steps = max(
            constraints.min_on_steps,
            max(1, round(30 / interval_minutes)),
        )
        active_indices = [
            step.index
            for step in flexibility_state.steps
            if step.pv_surplus_window_kwh > 0.0
            and step.available_storage_kwh > 0.0
            and step.expected_discharge_need_kwh > 0.0
        ]
        blocks = self._build_blocks_from_indices(
            active_indices=active_indices,
            flexibility_state=flexibility_state,
            useful_run_steps=useful_run_steps,
            interval_minutes=interval_minutes,
        )
        step_to_block_id: list[int | None] = [None for _ in flexibility_state.steps]
        for block in blocks:
            for index in range(block.start_index, block.end_index + 1):
                step_to_block_id[index] = block.block_id
        return PreheatSchedule(
            blocks=blocks,
            step_to_block_id=step_to_block_id,
            total_planned_charge_kwh=sum(block.planned_charge_kwh for block in blocks),
            diagnostics={
                "block_count": len(blocks),
                "merged_gap_tolerance_steps": 1,
            },
        )

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
    ) -> list[PreheatBlock]:
        if not active_indices:
            return []
        blocks: list[PreheatBlock] = []
        current_block_indices: list[int] = []
        block_id = 0
        gap_tolerance_steps = 1
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
            (flexibility_state.steps[index].expected_discharge_need_kwh for index in full_range_indices),
            default=0.0,
        )
        planned_charge_kwh = min(
            available_surplus_kwh,
            available_storage_kwh,
            expected_discharge_need_kwh if expected_discharge_need_kwh > 0.0 else available_storage_kwh,
        )
        max_hp_power_kw = max(
            (
                flexibility_state.steps[index].pv_surplus_forecast_kw
                for index in full_range_indices
            ),
            default=0.0,
        )
        minimum_useful_charge_kwh = max_hp_power_kw * useful_run_steps * dt_hours * 0.5
        if (
            len(full_range_indices) < useful_run_steps
            and planned_charge_kwh < minimum_useful_charge_kwh
        ):
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
