from __future__ import annotations

from collections.abc import Sequence
import math

from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
    LinearThermalControlModel,
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    PreheatBlock,
    PreheatSchedule,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    ThermalFlexibilityState,
    ThermalFlexibilityStep,
)


class SpaceHeatingFlexibilityAssessor:
    def assess(
        self,
        *,
        interval_minutes: int,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: Sequence[MpcHorizonStep],
        constraints: MpcConstraints,
    ) -> ThermalFlexibilityState:
        if not horizon:
            return ThermalFlexibilityState(diagnostics={"reason": "empty_horizon"})

        dt_hours = interval_minutes / 60.0
        window_steps = (
            constraints.pv_opportunity_window_steps
            if constraints.pv_opportunity_window_steps > 0
            else max(1, round(90 / interval_minutes))
        )
        no_heat_rollout = self._rollout_without_heating_states(
            control_model=control_model,
            initial_state=initial_state,
            horizon=horizon,
        )
        max_import_price = max(
            (float(step.import_price_eur_kwh) for step in horizon),
            default=0.0,
        )
        pv_surplus_forecast_kw = [
            max(
                0.0,
                float(step.pv_available_power_forecast_kw) - float(step.base_load_power_forecast_kw),
            )
            for step in horizon
        ]
        pv_surplus_window_kwh = [
            sum(pv_surplus_forecast_kw[index : index + window_steps]) * dt_hours
            for index in range(len(horizon))
        ]

        steps: list[ThermalFlexibilityStep] = []
        total_storage = 0.0
        total_discharge_need = 0.0
        for index, step in enumerate(horizon):
            no_heat_room_temp_c, no_heat_mass_temp_c = no_heat_rollout[index]
            comfort_headroom_c = max(0.0, float(step.temp_max_c) - no_heat_room_temp_c)
            room_band_c = max(float(step.temp_max_c) - float(step.temp_min_c), 1e-9)
            future_need_c = max(
                (
                    max(
                        0.0,
                        max(
                            float(future_step.temp_min_c),
                            float(future_step.target_temp_c or future_step.temp_min_c),
                        )
                        - no_heat_rollout[future_index][0],
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
            available_storage_kwh = max(
                0.0,
                (comfort_headroom_c / heat_gain_c)
                * max(float(step.hp_electric_power_forecast_kw), 1e-9)
                * dt_hours,
            )
            expected_discharge_need_kwh = max(
                0.0,
                (max(economic_target_c - no_heat_room_temp_c, 0.0) / heat_gain_c)
                * max(float(step.hp_electric_power_forecast_kw), 1e-9)
                * dt_hours,
            )
            total_storage += available_storage_kwh
            total_discharge_need += expected_discharge_need_kwh
            steps.append(
                ThermalFlexibilityStep(
                    index=index,
                    timestamp_utc=step.timestamp_utc,
                    temp_min_c=step.temp_min_c,
                    temp_max_c=step.temp_max_c,
                    economic_target_c=economic_target_c,
                    no_heat_room_temp_c=no_heat_room_temp_c,
                    no_heat_mass_temp_c=no_heat_mass_temp_c,
                    comfort_headroom_c=comfort_headroom_c,
                    available_storage_kwh=available_storage_kwh,
                    expected_discharge_need_kwh=expected_discharge_need_kwh,
                    pv_surplus_forecast_kw=pv_surplus_forecast_kw[index],
                    pv_surplus_window_kwh=pv_surplus_window_kwh[index],
                )
            )

        return ThermalFlexibilityState(
            steps=steps,
            total_available_storage_kwh=total_storage,
            total_expected_discharge_need_kwh=total_discharge_need,
            diagnostics={
                "window_steps": window_steps,
                "step_count": len(steps),
            },
        )

    def build_execution_targets(
        self,
        *,
        flexibility_state: ThermalFlexibilityState,
        schedule: PreheatSchedule,
    ) -> list[ExecutionTargetStep]:
        blocks_by_id = {block.block_id: block for block in schedule.blocks}
        cumulative_targets_by_block: dict[int, dict[int, float]] = {}
        for block in schedule.blocks:
            block_indices = list(range(block.start_index, block.end_index + 1))
            raw_weights = [
                max(flexibility_state.steps[index].pv_surplus_forecast_kw, 0.0)
                for index in block_indices
            ]
            total_weight = sum(raw_weights)
            if total_weight <= 0.0:
                weights = [1.0 / max(len(block_indices), 1) for _ in block_indices]
            else:
                weights = [weight / total_weight for weight in raw_weights]
            running_target = 0.0
            cumulative_targets_by_block[block.block_id] = {}
            for index, weight in zip(block_indices, weights, strict=False):
                running_target += block.planned_charge_kwh * weight
                cumulative_targets_by_block[block.block_id][index] = running_target
        targets: list[ExecutionTargetStep] = []
        for step in flexibility_state.steps:
            block_id = (
                schedule.step_to_block_id[step.index]
                if step.index < len(schedule.step_to_block_id)
                else None
            )
            block = blocks_by_id.get(block_id) if block_id is not None else None
            preheat_target_c = step.economic_target_c
            start_allowed_for_preheat = False
            start_reason_hint = "economic_target_only"
            max_preheat_target_c = step.economic_target_c
            remaining_budget_kwh = 0.0
            if block is not None:
                max_preheat_target_c = block.max_preheat_target_c
                preheat_target_c = max(step.economic_target_c, block.max_preheat_target_c)
                remaining_budget_kwh = max(block.planned_charge_kwh, 0.0)
                start_allowed_for_preheat = remaining_budget_kwh > 0.0
                start_reason_hint = "preheat_block_active"
                block_budget_share_kwh = max(
                    cumulative_targets_by_block[block.block_id].get(step.index, 0.0)
                    - cumulative_targets_by_block[block.block_id].get(step.index - 1, 0.0),
                    0.0,
                )
                block_cumulative_budget_target_kwh = cumulative_targets_by_block[
                    block.block_id
                ].get(step.index, 0.0)
            else:
                block_budget_share_kwh = 0.0
                block_cumulative_budget_target_kwh = 0.0
            targets.append(
                ExecutionTargetStep(
                    timestamp_utc=step.timestamp_utc,
                    economic_target_c=step.economic_target_c,
                    preheat_target_c=preheat_target_c,
                    active_preheat_block_id=block_id,
                    remaining_block_budget_kwh=remaining_budget_kwh,
                    block_budget_share_kwh=block_budget_share_kwh,
                    block_cumulative_budget_target_kwh=block_cumulative_budget_target_kwh,
                    max_preheat_target_c=max_preheat_target_c,
                    start_allowed_for_preheat=start_allowed_for_preheat,
                    start_reason_hint=start_reason_hint,
                )
            )
        return targets

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
            0.45,
            min(
                1.0,
                0.7 + (0.3 * future_need_score) - (0.45 * price_norm),
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

    @staticmethod
    def _rollout_without_heating_states(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: Sequence[MpcHorizonStep],
    ) -> list[tuple[float, float | None]]:
        if not horizon:
            return []

        states: list[tuple[float, float | None]] = []
        current_room_temp_c = initial_state.room_temp_c
        current_mass_temp_c = (
            initial_state.mass_temp_c if isinstance(initial_state, Rc2StateMpcInitialState) else None
        )
        current_q_heat_eff_kw = initial_state.q_heat_eff_kw
        for step in horizon:
            states.append((current_room_temp_c, current_mass_temp_c))
            current_q_heat_eff_kw = control_model.actuator_alpha * current_q_heat_eff_kw
            if isinstance(control_model, Rc2StateThermalControlModel):
                if current_mass_temp_c is None:
                    raise ValueError("Rc2StateThermalControlModel requires Rc2StateMpcInitialState")
                current_room_temp_c, current_mass_temp_c = control_model.predict_next_state(
                    room_temp_c=current_room_temp_c,
                    mass_temp_c=current_mass_temp_c,
                    outdoor_temp_c=step.outdoor_temp_c,
                    solar_gain_kw=step.solar_gain_kw,
                    solar_gain_mass_kw=float(step.solar_gain_mass_kw),
                    heating_effect_kw=current_q_heat_eff_kw,
                    occupied=step.occupied,
                    hour_sin=step.hour_sin,
                    hour_cos=step.hour_cos,
                )
            else:
                current_room_temp_c = control_model.predict_next_temperature(
                    room_temp_c=current_room_temp_c,
                    outdoor_temp_c=step.outdoor_temp_c,
                    solar_gain_kw=step.solar_gain_kw,
                    heating_effect_kw=current_q_heat_eff_kw,
                    occupied=step.occupied,
                )
        return states
