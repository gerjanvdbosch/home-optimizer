from __future__ import annotations

from collections.abc import Sequence
import math

from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
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
        control_model: Rc2StateThermalControlModel,
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
        expected_discharge_need_kwh = self._expected_discharge_need_profile_kwh(
            horizon=horizon,
            no_heat_rollout=no_heat_rollout,
            max_import_price=max_import_price,
            control_model=control_model,
            dt_hours=dt_hours,
        )

        steps: list[ThermalFlexibilityStep] = []
        total_storage = 0.0
        total_discharge_need = 0.0
        for index, step in enumerate(horizon):
            room_temp_c, mass_temp_c, q_heat_eff_kw = no_heat_rollout[index]
            no_heat_room_temp_c = room_temp_c
            no_heat_mass_temp_c = mass_temp_c
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
            room_mass_delta_c = (
                no_heat_room_temp_c - no_heat_mass_temp_c
                if no_heat_mass_temp_c is not None
                else 0.0
            )
            mass_reference_temp_c = (
                no_heat_mass_temp_c if no_heat_mass_temp_c is not None else no_heat_room_temp_c
            )
            mass_deficit_to_economic_target_c = max(
                economic_target_c - mass_reference_temp_c,
                0.0,
            )
            mass_deficit_to_preheat_target_c = max(
                float(step.temp_max_c) - mass_reference_temp_c,
                0.0,
            )
            band_c = max(float(step.temp_max_c) - float(step.temp_min_c), 1e-9)
            normalized_storage_soc = min(
                max((mass_reference_temp_c - float(step.temp_min_c)) / band_c, 0.0),
                1.0,
            )
            estimated_storage_soc_kwh = available_storage_kwh * normalized_storage_soc
            total_storage += available_storage_kwh
            total_discharge_need += expected_discharge_need_kwh[index]
            steps.append(
                ThermalFlexibilityStep(
                    index=index,
                    timestamp_utc=step.timestamp_utc,
                    temp_min_c=step.temp_min_c,
                    temp_max_c=step.temp_max_c,
                    economic_target_c=economic_target_c,
                    room_temp_c=no_heat_room_temp_c,
                    mass_temp_c=no_heat_mass_temp_c,
                    q_heat_eff_kw=q_heat_eff_kw,
                    no_heat_room_temp_c=no_heat_room_temp_c,
                    no_heat_mass_temp_c=no_heat_mass_temp_c,
                    room_mass_delta_c=room_mass_delta_c,
                    mass_deficit_to_economic_target_c=mass_deficit_to_economic_target_c,
                    mass_deficit_to_preheat_target_c=mass_deficit_to_preheat_target_c,
                    normalized_storage_soc=normalized_storage_soc,
                    estimated_storage_soc_kwh=estimated_storage_soc_kwh,
                    comfort_headroom_c=comfort_headroom_c,
                    available_storage_kwh=available_storage_kwh,
                    expected_discharge_need_kwh=expected_discharge_need_kwh[index],
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

    def _expected_discharge_need_profile_kwh(
        self,
        *,
        horizon: Sequence[MpcHorizonStep],
        no_heat_rollout: Sequence[tuple[float, float | None, float]],
        max_import_price: float,
        control_model: Rc2StateThermalControlModel,
        dt_hours: float,
    ) -> list[float]:
        if not horizon:
            return []

        future_need_profile: list[float] = [0.0 for _ in horizon]
        for index, step in enumerate(horizon):
            economic_target_c = self._economic_target_c(
                step=step,
                future_need_score=0.0,
                max_import_price=max_import_price,
            )
            heat_gain_c = max(
                self._one_step_room_heat_gain_c(
                    control_model=control_model,
                    step=step,
                ),
                0.05,
            )
            hp_power_kw = max(float(step.hp_electric_power_forecast_kw), 1e-9)
            running_need_kwh = 0.0
            for future_index in range(index, len(horizon)):
                future_room_temp_c = no_heat_rollout[future_index][0]
                future_target_c = self._economic_target_c(
                    step=horizon[future_index],
                    future_need_score=0.0,
                    max_import_price=max_import_price,
                )
                deficit_c = max(future_target_c - future_room_temp_c, 0.0)
                if deficit_c <= 0.0:
                    continue
                step_need_kwh = (deficit_c / heat_gain_c) * hp_power_kw * dt_hours
                lookahead_decay = 1.0 / (1.0 + 0.15 * (future_index - index))
                running_need_kwh += step_need_kwh * lookahead_decay
            future_need_profile[index] = max(running_need_kwh, 0.0)
        return future_need_profile

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
                start_reason_hint = (
                    "post_solar_hold_needed"
                    if block.post_solar_no_heat_drops_below_economic_target
                    or block.post_solar_no_heat_drops_below_temp_min
                    else "post_solar_hold_ok"
                )
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
                    storage_target_kwh=block.planned_charge_kwh if block is not None else 0.0,
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
        control_model: Rc2StateThermalControlModel,
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
        control_model: Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: Sequence[MpcHorizonStep],
    ) -> list[tuple[float, float | None, float]]:
        if not horizon:
            return []

        states: list[tuple[float, float | None, float]] = []
        current_room_temp_c = initial_state.room_temp_c
        current_mass_temp_c = (
            initial_state.mass_temp_c if isinstance(initial_state, Rc2StateMpcInitialState) else None
        )
        current_q_heat_eff_kw = initial_state.q_heat_eff_kw
        for step in horizon:
            states.append((current_room_temp_c, current_mass_temp_c, current_q_heat_eff_kw))
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
