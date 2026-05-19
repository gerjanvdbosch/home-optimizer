from __future__ import annotations

from collections.abc import Sequence

from home_optimizer.features.mpc.models import (
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc_new.models import (
    IntentPlanningState,
    IntentPlanningStep,
)


class IntentPlanningAssessor:
    def assess(
        self,
        *,
        interval_minutes: int,
        control_model: Rc2StateThermalControlModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: Sequence[MpcHorizonStep],
        constraints: MpcConstraints,
    ) -> IntentPlanningState:
        if not horizon:
            return IntentPlanningState(diagnostics={"reason": "empty_horizon"})

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
                float(step.pv_available_power_forecast_kw)
                - float(step.base_load_power_forecast_kw),
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
        post_solar_summaries = self._post_solar_no_heat_summaries(
            horizon=horizon,
            no_heat_rollout=no_heat_rollout,
        )

        steps: list[IntentPlanningStep] = []
        total_storage = 0.0
        total_discharge_need = 0.0
        for index, step in enumerate(horizon):
            room_temp_c, mass_temp_c, q_heat_eff_kw = no_heat_rollout[index]
            economic_target_c = self._economic_target_c(
                step=step,
                future_need_score=self._future_need_score(
                    index=index,
                    horizon=horizon,
                    no_heat_rollout=no_heat_rollout,
                ),
                max_import_price=max_import_price,
            )
            available_storage_kwh = self._available_storage_kwh(
                step=step,
                control_model=control_model,
                room_temp_c=room_temp_c,
                dt_hours=dt_hours,
            )
            post_solar_summary = post_solar_summaries[index]
            total_storage += available_storage_kwh
            total_discharge_need += expected_discharge_need_kwh[index]
            steps.append(
                IntentPlanningStep(
                    index=index,
                    timestamp_utc=step.timestamp_utc,
                    temp_min_c=step.temp_min_c,
                    temp_max_c=step.temp_max_c,
                    economic_target_c=economic_target_c,
                    room_temp_c=room_temp_c,
                    mass_temp_c=mass_temp_c,
                    q_heat_eff_kw=q_heat_eff_kw,
                    no_heat_room_temp_c=room_temp_c,
                    no_heat_mass_temp_c=mass_temp_c,
                    available_storage_kwh=available_storage_kwh,
                    expected_discharge_need_kwh=expected_discharge_need_kwh[index],
                    pv_surplus_forecast_kw=pv_surplus_forecast_kw[index],
                    pv_surplus_window_kwh=pv_surplus_window_kwh[index],
                    post_solar_no_heat_min_temp_c=post_solar_summary["min_room_temp_c"],
                    post_solar_no_heat_end_temp_c=post_solar_summary["end_room_temp_c"],
                    post_solar_no_heat_drops_below_economic_target=bool(
                        post_solar_summary["drops_below_economic"]
                    ),
                    post_solar_no_heat_drops_below_temp_min=bool(
                        post_solar_summary["drops_below_temp_min"]
                    ),
                )
            )

        return IntentPlanningState(
            steps=steps,
            total_available_storage_kwh=total_storage,
            total_expected_discharge_need_kwh=total_discharge_need,
            diagnostics={
                "window_steps": window_steps,
                "step_count": len(steps),
            },
        )

    def _post_solar_no_heat_summaries(
        self,
        *,
        horizon: Sequence[MpcHorizonStep],
        no_heat_rollout: Sequence[tuple[float, float | None, float]],
    ) -> list[dict[str, float | bool]]:
        summaries: list[dict[str, float | bool]] = []
        for index, _step in enumerate(horizon):
            min_room_temp_c = no_heat_rollout[index][0]
            end_room_temp_c = no_heat_rollout[index][0]
            drops_below_economic = False
            drops_below_temp_min = False
            for future_index in range(index, len(horizon)):
                future_room_temp_c = no_heat_rollout[future_index][0]
                end_room_temp_c = future_room_temp_c
                min_room_temp_c = min(min_room_temp_c, future_room_temp_c)
                if future_room_temp_c < float(
                    horizon[future_index].economic_target_c
                    or horizon[future_index].temp_min_c
                ):
                    drops_below_economic = True
                if future_room_temp_c < float(horizon[future_index].temp_min_c):
                    drops_below_temp_min = True
            summaries.append(
                {
                    "min_room_temp_c": min_room_temp_c,
                    "end_room_temp_c": end_room_temp_c,
                    "drops_below_economic": drops_below_economic,
                    "drops_below_temp_min": drops_below_temp_min,
                }
            )
        return summaries

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

    @staticmethod
    def _future_need_score(
        *,
        index: int,
        horizon: Sequence[MpcHorizonStep],
        no_heat_rollout: Sequence[tuple[float, float | None, float]],
    ) -> float:
        step = horizon[index]
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
        return min(future_need_c / room_band_c, 1.0)

    def _available_storage_kwh(
        self,
        *,
        step: MpcHorizonStep,
        control_model: Rc2StateThermalControlModel,
        room_temp_c: float,
        dt_hours: float,
    ) -> float:
        comfort_headroom_c = max(0.0, float(step.temp_max_c) - room_temp_c)
        heat_gain_c = max(
            self._one_step_room_heat_gain_c(
                control_model=control_model,
                step=step,
            ),
            0.05,
        )
        return max(
            0.0,
            (comfort_headroom_c / heat_gain_c)
            * max(float(step.hp_electric_power_forecast_kw), 1e-9)
            * dt_hours,
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
            initial_state.mass_temp_c
            if isinstance(initial_state, Rc2StateMpcInitialState)
            else None
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
