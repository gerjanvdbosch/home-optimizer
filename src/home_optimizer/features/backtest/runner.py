from __future__ import annotations

from datetime import datetime
from typing import Any

from home_optimizer.features.backtest.models import (
    MpcBacktestPvDiagnostics,
    MpcBacktestResult,
    MpcBacktestStepResult,
    MpcBacktestSummary,
)
from home_optimizer.features.mpc.controller_service import SpaceHeatingMpcControllerService
from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
    HeatPumpSequencerState,
    LinearThermalControlModel,
    MpcConstraints,
    MpcControllerRequest,
    MpcControlMode,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveBreakdown,
    MpcObjectiveWeights,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc_new.controller_service import IntentAwareMpcControllerService
from home_optimizer.features.mpc_new.models import (
    IntentAwareMpcControllerRequest,
    RunExecutionState,
    RunIntentExecutionTargetStep,
    RunIntentPlan,
)


class SpaceHeatingMpcBacktestRunner:
    def __init__(
        self,
        *,
        controller: SpaceHeatingMpcControllerService
        | IntentAwareMpcControllerService
        | None = None,
    ) -> None:
        self.controller = controller or SpaceHeatingMpcControllerService()

    def run(
        self,
        *,
        model_id: str,
        model_type: str,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        timeline: list[MpcHorizonStep],
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        interval_minutes: int,
        horizon_steps: int,
        constraints: MpcConstraints | None = None,
        objective_weights: MpcObjectiveWeights | None = None,
        max_solver_seconds: float | None = None,
        historical_hp_on_by_timestamp: dict[datetime, bool] | None = None,
        historical_energy_cost_by_timestamp: dict[datetime, float] | None = None,
        exogenous_mode: str = "perfect_foresight",
        control_mode: MpcControlMode = "hierarchical_preheat",
        forecast_replay_provider: Any | None = None,
    ) -> MpcBacktestResult:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")
        if len(timeline) < 2:
            raise ValueError("timeline must contain at least two steps for closed-loop replay")

        resolved_constraints = constraints or MpcConstraints()
        resolved_weights = objective_weights or MpcObjectiveWeights()
        current_state = initial_state
        historical_q_heat_eff_kw = initial_state.q_heat_eff_kw
        step_results: list[MpcBacktestStepResult] = []
        infeasible_count = 0
        total_solver_runtime_seconds = 0.0
        slack_usage_count = 0
        solver_cumulative_objective_breakdown = MpcObjectiveBreakdown()
        executed_path_objective_breakdown = MpcObjectiveBreakdown()
        historical_hp_on_by_timestamp = historical_hp_on_by_timestamp or {}
        historical_energy_cost_by_timestamp = historical_energy_cost_by_timestamp or {}
        missing_forecast_count = 0
        forecast_coverage_ratio = 1.0
        current_sequencer_state = HeatPumpSequencerState()
        current_run_execution_state = RunExecutionState()
        previous_intent_plan: RunIntentPlan | None = None

        for index in range(len(timeline) - 1):
            realized_horizon = timeline[index : index + horizon_steps]
            horizon = realized_horizon
            if not horizon:
                break
            forecast_issue_time_utc = horizon[0].timestamp_utc
            forecast_age_minutes = 0.0
            if forecast_replay_provider is not None:
                replay_horizon = forecast_replay_provider.get_forecast_horizon(
                    horizon[0].timestamp_utc,
                    len(horizon),
                    interval_minutes,
                )
                forecast_issue_time_utc = replay_horizon.forecast_issue_time_utc
                forecast_age_minutes = replay_horizon.forecast_age_minutes
                forecast_coverage_ratio = min(
                    forecast_coverage_ratio,
                    replay_horizon.forecast_coverage_ratio,
                )
                missing_forecast_count += replay_horizon.missing_forecast_count
                horizon = self._merge_replay_horizon(
                    realized_horizon=realized_horizon,
                    replay_horizon=replay_horizon.horizon,
                )
            current_forecast_step = horizon[0]
            current_realized_step = timeline[index]

            if isinstance(self.controller, IntentAwareMpcControllerService):
                request = IntentAwareMpcControllerRequest(
                    interval_minutes=interval_minutes,
                    horizon=horizon,
                    control_mode=control_mode,
                    run_execution_state=current_run_execution_state,
                    previous_intent_plan=previous_intent_plan,
                    constraints=resolved_constraints,
                    objective_weights=resolved_weights,
                    max_solver_seconds=max_solver_seconds,
                )
            else:
                request = MpcControllerRequest(
                    interval_minutes=interval_minutes,
                    horizon=horizon,
                    control_mode=control_mode,
                    sequencer_state=current_sequencer_state,
                    constraints=resolved_constraints,
                    objective_weights=resolved_weights,
                    max_solver_seconds=max_solver_seconds,
                )
            plan = self.controller.plan(
                request,
                control_model=control_model,
                initial_state=current_state,
                horizon=horizon,
            )
            solver_cumulative_objective_breakdown = self._add_objective_breakdowns(
                solver_cumulative_objective_breakdown,
                plan.objective_breakdown,
            )

            if not plan.feasible or not plan.steps:
                infeasible_count += 1
                applied_hp_on = current_state.hp_on
                start = False
                stop = False
                solve_time_seconds = plan.solve_time_seconds
                slack_low_c = 0.0
                slack_high_c = 0.0
                first_step = None
            else:
                first_step = plan.steps[0]
                applied_hp_on = first_step.hp_on
                start = first_step.start
                stop = first_step.stop
                solve_time_seconds = plan.solve_time_seconds
                slack_low_c = first_step.slack_low_c
                slack_high_c = first_step.slack_high_c
                if first_step.slack_low_c > 0.0 or first_step.slack_high_c > 0.0:
                    slack_usage_count += 1

            next_step = timeline[index + 1]
            q_heat_eff_kw = self._apply_heat_actuator(
                control_model=control_model,
                previous_q_heat_eff_kw=current_state.q_heat_eff_kw,
                commanded_hp_on=applied_hp_on,
                current_step=current_realized_step,
            )
            predicted_next_temp, next_state = self._predict_next_state(
                control_model=control_model,
                current_state=current_state,
                current_step=current_realized_step,
                heating_effect_kw=q_heat_eff_kw,
            )
            total_solver_runtime_seconds += solve_time_seconds or 0.0

            historical_hp_on = historical_hp_on_by_timestamp.get(
                current_realized_step.timestamp_utc,
                current_realized_step.effective_heating_kw_forecast > 0.0,
            )
            historical_q_heat_eff_kw = self._apply_heat_actuator(
                control_model=control_model,
                previous_q_heat_eff_kw=historical_q_heat_eff_kw,
                commanded_hp_on=historical_hp_on,
                current_step=current_realized_step,
            )

            step_results.append(
                MpcBacktestStepResult(
                    timestamp_utc=current_realized_step.timestamp_utc,
                    forecast_issue_time_utc=forecast_issue_time_utc,
                    forecast_age_minutes=forecast_age_minutes,
                    mpc_hp_on=applied_hp_on,
                    historical_hp_on=historical_hp_on,
                    start=start,
                    stop=stop,
                    planned_room_temp_c=(
                        first_step.predicted_room_temp_c
                        if plan.feasible and plan.steps
                        else current_state.room_temp_c
                    ),
                    useful_preheat_target_c=(
                        first_step.useful_preheat_target_c
                        if plan.feasible and plan.steps
                        else float(
                            current_forecast_step.target_temp_c or current_forecast_step.temp_min_c
                        )
                    ),
                    preheat_active=(
                        first_step.preheat_active
                        if plan.feasible and plan.steps
                        else bool(current_forecast_step.preheat_active)
                    ),
                    preheat_block_id=(
                        first_step.preheat_block_id
                        if plan.feasible and plan.steps
                        else current_forecast_step.preheat_block_id
                    ),
                    preheat_budget_share_kwh=(
                        first_step.preheat_budget_share_kwh
                        if plan.feasible and plan.steps
                        else float(current_forecast_step.preheat_budget_share_kwh)
                    ),
                    preheat_charge_kwh=(
                        first_step.preheat_charge_kwh if plan.feasible and plan.steps else 0.0
                    ),
                    preheat_opportunity_score=(
                        first_step.preheat_opportunity_score
                        if plan.feasible and plan.steps
                        else float(current_forecast_step.preheat_opportunity_score)
                    ),
                    q_heat_eff_kw=q_heat_eff_kw,
                    historical_q_heat_eff_kw=historical_q_heat_eff_kw,
                    hp_electric_power_kw=current_forecast_step.hp_electric_power_forecast_kw,
                    pv_forecast_kw=current_forecast_step.pv_available_power_forecast_kw,
                    pv_realized_kw=float(
                        current_realized_step.pv_available_power_realized_kw or 0.0
                    ),
                    solar_irradiance_forecast_wm2=current_forecast_step.solar_irradiance_forecast_w_m2,
                    solar_irradiance_realized_wm2=current_realized_step.solar_irradiance_realized_w_m2,
                    solar_gain_forecast_kw=current_forecast_step.solar_gain_kw,
                    solar_gain_realized_kw=float(
                        current_realized_step.solar_gain_realized_kw or 0.0
                    ),
                    base_load_forecast_kw=current_forecast_step.base_load_power_forecast_kw,
                    base_load_realized_kw=float(
                        current_realized_step.base_load_power_realized_kw or 0.0
                    ),
                    pv_surplus_forecast_kw=max(
                        current_forecast_step.pv_available_power_forecast_kw
                        - current_forecast_step.base_load_power_forecast_kw,
                        0.0,
                    ),
                    pv_surplus_realized_kw=max(
                        float(current_realized_step.pv_available_power_realized_kw or 0.0)
                        - float(current_realized_step.base_load_power_realized_kw or 0.0),
                        0.0,
                    ),
                    predicted_next_room_temp_c=predicted_next_temp,
                    simulated_next_room_temp_c=predicted_next_temp,
                    historical_next_room_temp_c=next_step.realized_room_temp_c,
                    temp_min_c=next_step.temp_min_c,
                    temp_max_c=next_step.temp_max_c,
                    slack_low_c=slack_low_c,
                    slack_high_c=slack_high_c,
                    price_eur_kwh=current_forecast_step.import_price_eur_kwh,
                    estimated_mpc_energy_cost_eur=self._site_energy_cost(
                        current_step=current_realized_step,
                        hp_on=applied_hp_on,
                        interval_minutes=interval_minutes,
                    ),
                    estimated_historical_energy_cost_eur=self._site_energy_cost(
                        current_step=current_realized_step,
                        hp_on=historical_hp_on,
                        interval_minutes=interval_minutes,
                        override_cost=historical_energy_cost_by_timestamp.get(
                            current_realized_step.timestamp_utc
                        ),
                    ),
                    solve_time_seconds=solve_time_seconds,
                    feasible=plan.feasible,
                )
            )
            executed_path_objective_breakdown = self._add_objective_breakdowns(
                executed_path_objective_breakdown,
                self._executed_step_objective_breakdown(
                    current_step=current_forecast_step,
                    planned_room_temp_c=(
                        first_step.predicted_room_temp_c
                        if plan.feasible and plan.steps
                        else current_state.room_temp_c
                    ),
                    useful_preheat_target_c=(
                        first_step.useful_preheat_target_c
                        if plan.feasible and plan.steps
                        else float(
                            current_forecast_step.target_temp_c or current_forecast_step.temp_min_c
                        )
                    ),
                    hp_on=applied_hp_on,
                    q_heat_eff_kw=q_heat_eff_kw,
                    start=start,
                    slack_low_c=slack_low_c,
                    slack_high_c=slack_high_c,
                    interval_minutes=interval_minutes,
                    objective_weights=resolved_weights,
                ),
            )

            current_state = self._advance_state(
                next_state=next_state.model_copy(update={"q_heat_eff_kw": q_heat_eff_kw}),
                hp_on=applied_hp_on,
            )
            if isinstance(self.controller, IntentAwareMpcControllerService):
                previous_intent_plan = plan.run_intent_plan
                executed_target = (
                    plan.execution_targets[0]
                    if plan.execution_targets
                    else self._fallback_intent_target(current_forecast_step, first_step)
                )
                current_run_execution_state = self.controller.advance_execution_state(
                    state=current_run_execution_state,
                    executed_step=current_forecast_step,
                    executed_target=executed_target,
                    executed_hp_on=applied_hp_on,
                    interval_minutes=interval_minutes,
                    preheat_charge_kwh=first_step.preheat_charge_kwh
                    if first_step is not None
                    else 0.0,
                )
            else:
                executed_target = ExecutionTargetStep(
                    timestamp_utc=current_forecast_step.timestamp_utc,
                    economic_target_c=float(
                        first_step.economic_target_c
                        if first_step is not None
                        else current_forecast_step.economic_target_c
                        or current_forecast_step.temp_min_c
                    ),
                    preheat_target_c=float(
                        first_step.useful_preheat_target_c
                        if first_step is not None
                        else current_forecast_step.max_preheat_target_c
                        or current_forecast_step.temp_min_c
                    ),
                    active_preheat_block_id=(
                        first_step.preheat_block_id
                        if first_step is not None
                        else current_forecast_step.preheat_block_id
                    ),
                    remaining_block_budget_kwh=0.0,
                    block_budget_share_kwh=(
                        first_step.preheat_budget_share_kwh
                        if first_step is not None
                        else float(current_forecast_step.preheat_budget_share_kwh)
                    ),
                    block_cumulative_budget_target_kwh=float(
                        current_forecast_step.preheat_block_cumulative_target_kwh
                    ),
                    storage_target_kwh=float(current_forecast_step.preheat_block_budget_kwh),
                    max_preheat_target_c=float(
                        current_forecast_step.max_preheat_target_c
                        or current_forecast_step.temp_max_c
                    ),
                    start_allowed_for_preheat=bool(
                        first_step.hp_start_allowed
                        if first_step is not None
                        else current_forecast_step.hp_start_allowed
                    ),
                    start_reason_hint=(
                        first_step.start_reason
                        if first_step is not None
                        else current_forecast_step.start_reason_hint
                    ),
                    sequencer_mode=(
                        first_step.sequencer_mode
                        if first_step is not None
                        else current_forecast_step.sequencer_mode
                    ),
                    active_run_id=(
                        first_step.active_run_id
                        if first_step is not None
                        else current_forecast_step.active_run_id
                    ),
                    hp_must_be_on=bool(
                        first_step.hp_must_be_on
                        if first_step is not None
                        else current_forecast_step.hp_must_be_on
                    ),
                    hp_must_be_off=bool(
                        first_step.hp_must_be_off
                        if first_step is not None
                        else current_forecast_step.hp_must_be_off
                    ),
                    hp_start_allowed=bool(
                        first_step.hp_start_allowed
                        if first_step is not None
                        else current_forecast_step.hp_start_allowed
                    ),
                    stop_reason_hint=(
                        first_step.stop_reason
                        if first_step is not None
                        else current_forecast_step.stop_reason_hint
                    ),
                    committed_on_until_utc=(
                        first_step.committed_on_until_utc
                        if first_step is not None
                        else current_forecast_step.committed_on_until_utc
                    ),
                    locked_off_until_utc=(
                        first_step.locked_off_until_utc
                        if first_step is not None
                        else current_forecast_step.locked_off_until_utc
                    ),
                    starts_used_in_block=(
                        first_step.starts_used_in_block
                        if first_step is not None
                        else current_forecast_step.starts_used_in_block
                    ),
                    run_budget_used_kwh=(
                        first_step.run_budget_used_kwh
                        if first_step is not None
                        else current_forecast_step.run_budget_used_kwh
                    ),
                    starts_blocked_by_lockout=bool(
                        first_step.starts_blocked_by_lockout
                        if first_step is not None
                        else current_forecast_step.starts_blocked_by_lockout
                    ),
                    starts_blocked_by_max_starts=bool(
                        first_step.starts_blocked_by_max_starts
                        if first_step is not None
                        else current_forecast_step.starts_blocked_by_max_starts
                    ),
                    starts_blocked_by_existing_commitment=bool(
                        first_step.starts_blocked_by_existing_commitment
                        if first_step is not None
                        else current_forecast_step.starts_blocked_by_existing_commitment
                    ),
                    stop_conditions=[],
                )
                if hasattr(self.controller, "advance_sequencer_state"):
                    current_sequencer_state = self.controller.advance_sequencer_state(
                        request_key=None,
                        state=current_sequencer_state,
                        executed_step=current_forecast_step,
                        executed_target=executed_target,
                        executed_hp_on=applied_hp_on,
                        interval_minutes=interval_minutes,
                        preheat_charge_kwh=first_step.preheat_charge_kwh
                        if first_step is not None
                        else 0.0,
                    )

        if step_results:
            last_step = step_results[-1]
            executed_path_objective_breakdown = executed_path_objective_breakdown.model_copy(
                update={
                    "terminal": executed_path_objective_breakdown.terminal
                    + (
                        resolved_weights.terminal
                        * max(
                            last_step.useful_preheat_target_c
                            - last_step.predicted_next_room_temp_c,
                            0.0,
                        )
                    )
                }
            )
            executed_path_objective_breakdown = executed_path_objective_breakdown.model_copy(
                update={
                    "preheat_budget_shortfall": self._executed_preheat_budget_shortfall_cost(
                        step_results=step_results,
                        interval_minutes=interval_minutes,
                        objective_weights=resolved_weights,
                    )
                }
            )

        average_solver_runtime = (
            total_solver_runtime_seconds / len(step_results) if step_results else 0.0
        )
        return MpcBacktestResult(
            exogenous_mode=exogenous_mode,
            control_mode=control_mode,
            missing_forecast_count=missing_forecast_count,
            forecast_coverage_ratio=forecast_coverage_ratio if step_results else 1.0,
            model_id=model_id,
            model_type=model_type,
            start_time_utc=timeline[0].timestamp_utc,
            end_time_utc=timeline[-1].timestamp_utc,
            interval_minutes=interval_minutes,
            horizon_steps=horizon_steps,
            step_results=step_results,
            mpc_summary=self._summarize_step_results(
                step_results,
                interval_minutes=interval_minutes,
                mode="mpc",
                q_heat_eff_active_threshold_kw=resolved_weights.q_heat_eff_active_threshold_kw,
                infeasible_count=infeasible_count,
                slack_usage_count=slack_usage_count,
                average_solver_runtime_seconds=average_solver_runtime,
            ),
            historical_summary=self._summarize_step_results(
                step_results,
                interval_minutes=interval_minutes,
                mode="historical",
                q_heat_eff_active_threshold_kw=resolved_weights.q_heat_eff_active_threshold_kw,
            ),
            pv_diagnostics=self._pv_diagnostics(
                step_results,
                interval_minutes=interval_minutes,
            ),
            mpc_objective_breakdown=executed_path_objective_breakdown,
            solver_objective_breakdown=solver_cumulative_objective_breakdown,
            total_solver_runtime_seconds=total_solver_runtime_seconds,
        )

    @staticmethod
    def _fallback_intent_target(
        current_forecast_step: MpcHorizonStep,
        first_step,
    ) -> RunIntentExecutionTargetStep:
        return RunIntentExecutionTargetStep(
            timestamp_utc=current_forecast_step.timestamp_utc,
            active_intent_id=None,
            active_run_id=(
                first_step.active_run_id
                if first_step is not None
                else current_forecast_step.active_run_id
            ),
            eligible_intent_id=None,
            hp_must_be_on=bool(
                first_step.hp_must_be_on
                if first_step is not None
                else current_forecast_step.hp_must_be_on
            ),
            hp_must_be_off=bool(
                first_step.hp_must_be_off
                if first_step is not None
                else current_forecast_step.hp_must_be_off
            ),
            hp_start_allowed=bool(
                first_step.hp_start_allowed
                if first_step is not None
                else current_forecast_step.hp_start_allowed
            ),
            target_charge_remaining_kwh=0.0,
            max_preheat_target_c=float(
                current_forecast_step.max_preheat_target_c or current_forecast_step.temp_max_c
            ),
            start_reason_hint=(
                first_step.start_reason
                if first_step is not None
                else current_forecast_step.start_reason_hint
            ),
            stop_reason_hint=(
                first_step.stop_reason
                if first_step is not None
                else current_forecast_step.stop_reason_hint
            ),
            committed_on_until_utc=(
                first_step.committed_on_until_utc
                if first_step is not None
                else current_forecast_step.committed_on_until_utc
            ),
            locked_off_until_utc=(
                first_step.locked_off_until_utc
                if first_step is not None
                else current_forecast_step.locked_off_until_utc
            ),
            mode=(
                first_step.sequencer_mode
                if first_step is not None
                else current_forecast_step.sequencer_mode
            ),
            starts_blocked_no_intent=not bool(
                first_step.hp_start_allowed
                if first_step is not None
                else current_forecast_step.hp_start_allowed
            ),
            comfort_fallback_allowed=True,
        )

    @staticmethod
    def _merge_replay_horizon(
        *,
        realized_horizon: list[MpcHorizonStep],
        replay_horizon: list[MpcHorizonStep],
    ) -> list[MpcHorizonStep]:
        merged: list[MpcHorizonStep] = []
        for realized_step, replay_step in zip(realized_horizon, replay_horizon, strict=False):
            merged.append(
                realized_step.model_copy(
                    update={
                        "outdoor_temp_c": replay_step.outdoor_temp_c,
                        "solar_gain_kw": replay_step.solar_gain_kw,
                        "solar_gain_mass_kw": replay_step.solar_gain_mass_kw,
                        "solar_irradiance_forecast_w_m2": (
                            replay_step.solar_irradiance_forecast_w_m2
                        ),
                        "pv_available_power_forecast_kw": (
                            replay_step.pv_available_power_forecast_kw
                        ),
                        "base_load_power_forecast_kw": replay_step.base_load_power_forecast_kw,
                        "price_eur_kwh": replay_step.price_eur_kwh,
                        "import_price_eur_kwh": replay_step.import_price_eur_kwh,
                        "export_price_eur_kwh": replay_step.export_price_eur_kwh,
                    }
                )
            )
        return merged

    @staticmethod
    def _add_objective_breakdowns(
        left: MpcObjectiveBreakdown,
        right: MpcObjectiveBreakdown,
    ) -> MpcObjectiveBreakdown:
        return MpcObjectiveBreakdown(
            comfort_low=left.comfort_low + right.comfort_low,
            active_comfort_high=left.active_comfort_high + right.active_comfort_high,
            passive_comfort_high=left.passive_comfort_high + right.passive_comfort_high,
            tracking_under_target=(left.tracking_under_target + right.tracking_under_target),
            tracking_over_target=(left.tracking_over_target + right.tracking_over_target),
            unnecessary_heating=(left.unnecessary_heating + right.unnecessary_heating),
            terminal=left.terminal + right.terminal,
            start=left.start + right.start,
            runtime=left.runtime + right.runtime,
            energy_cost=left.energy_cost + right.energy_cost,
            pv_self_consumption_reward=(
                left.pv_self_consumption_reward + right.pv_self_consumption_reward
            ),
            captured_pv_kwh=left.captured_pv_kwh + right.captured_pv_kwh,
            preheat_budget_shortfall=(
                left.preheat_budget_shortfall + right.preheat_budget_shortfall
            ),
        )

    @staticmethod
    def _executed_step_objective_breakdown(
        *,
        current_step: MpcHorizonStep,
        planned_room_temp_c: float,
        useful_preheat_target_c: float,
        hp_on: bool,
        q_heat_eff_kw: float,
        start: bool,
        slack_low_c: float,
        slack_high_c: float,
        interval_minutes: int,
        objective_weights: MpcObjectiveWeights,
    ) -> MpcObjectiveBreakdown:
        tracking_under_c = max(useful_preheat_target_c - planned_room_temp_c, 0.0)
        tracking_over_c = max(planned_room_temp_c - useful_preheat_target_c, 0.0)
        active_heating = hp_on or (q_heat_eff_kw > objective_weights.q_heat_eff_active_threshold_kw)
        pv_surplus_kw = max(
            0.0,
            current_step.pv_available_power_forecast_kw - current_step.base_load_power_forecast_kw,
        )
        pv_self_consumable_kw = min(
            current_step.hp_electric_power_forecast_kw * float(int(hp_on)),
            pv_surplus_kw,
        )
        dt_hours = interval_minutes / 60.0
        captured_pv_kwh = pv_self_consumable_kw * dt_hours
        return MpcObjectiveBreakdown(
            comfort_low=(objective_weights.comfort_low * dt_hours * slack_low_c),
            active_comfort_high=(
                float(objective_weights.active_comfort_high or 0.0)
                * dt_hours
                * slack_high_c
                * float(int(active_heating))
            ),
            passive_comfort_high=(
                objective_weights.passive_comfort_high
                * dt_hours
                * slack_high_c
                * float(int(not active_heating))
            ),
            tracking_under_target=objective_weights.tracking_under_target * tracking_under_c,
            tracking_over_target=objective_weights.tracking_over_target * tracking_over_c,
            unnecessary_heating=(
                objective_weights.unnecessary_heating * tracking_over_c * float(int(active_heating))
            ),
            terminal=0.0,
            start=objective_weights.start * float(int(start)),
            runtime=objective_weights.runtime * float(int(hp_on)),
            energy_cost=objective_weights.energy
            * SpaceHeatingMpcBacktestRunner._site_energy_cost_delta(
                current_step=current_step,
                hp_on=hp_on,
                interval_minutes=interval_minutes,
            ),
            pv_self_consumption_reward=(objective_weights.pv_self_consumption * captured_pv_kwh),
            captured_pv_kwh=captured_pv_kwh,
            preheat_budget_shortfall=0.0,
        )

    @staticmethod
    def _predict_next_state(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        current_state: MpcInitialState | Rc2StateMpcInitialState,
        current_step: MpcHorizonStep,
        heating_effect_kw: float,
    ) -> tuple[float, MpcInitialState | Rc2StateMpcInitialState]:
        if isinstance(control_model, Rc2StateThermalControlModel):
            if not isinstance(current_state, Rc2StateMpcInitialState):
                raise ValueError("2-state control model requires a 2-state MPC initial state")
            next_room_temp_c, next_mass_temp_c = control_model.predict_next_state(
                room_temp_c=current_state.room_temp_c,
                mass_temp_c=current_state.mass_temp_c,
                outdoor_temp_c=current_step.outdoor_temp_c,
                solar_gain_kw=current_step.solar_gain_kw,
                solar_gain_mass_kw=float(current_step.solar_gain_mass_kw),
                heating_effect_kw=heating_effect_kw,
                occupied=current_step.occupied,
                hour_sin=current_step.hour_sin,
                hour_cos=current_step.hour_cos,
            )
            return next_room_temp_c, current_state.model_copy(
                update={
                    "room_temp_c": next_room_temp_c,
                    "mass_temp_c": next_mass_temp_c,
                }
            )

        next_temp_c = control_model.predict_next_temperature(
            room_temp_c=current_state.room_temp_c,
            outdoor_temp_c=current_step.outdoor_temp_c,
            solar_gain_kw=current_step.solar_gain_kw,
            heating_effect_kw=heating_effect_kw,
            occupied=current_step.occupied,
        )
        return next_temp_c, current_state.model_copy(update={"room_temp_c": next_temp_c})

    @staticmethod
    def _advance_state(
        *,
        next_state: MpcInitialState | Rc2StateMpcInitialState,
        hp_on: bool,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
        if hp_on:
            on_steps = next_state.on_steps + 1 if next_state.hp_on else 1
            return next_state.model_copy(
                update={
                    "hp_on": True,
                    "on_steps": on_steps,
                    "off_steps": 0,
                }
            )
        off_steps = next_state.off_steps + 1 if not next_state.hp_on else 1
        return next_state.model_copy(
            update={
                "hp_on": False,
                "on_steps": 0,
                "off_steps": off_steps,
            }
        )

    @staticmethod
    def _apply_heat_actuator(
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel,
        previous_q_heat_eff_kw: float,
        commanded_hp_on: bool,
        current_step: MpcHorizonStep,
    ) -> float:
        return float(
            (control_model.actuator_alpha * previous_q_heat_eff_kw)
            + (
                (1.0 - control_model.actuator_alpha)
                * current_step.effective_heating_kw_forecast
                * float(int(commanded_hp_on))
            )
        )

    @staticmethod
    def _site_energy_cost(
        *,
        current_step: MpcHorizonStep,
        hp_on: bool,
        interval_minutes: int,
        override_cost: float | None = None,
    ) -> float:
        if override_cost is not None:
            return float(override_cost)
        net_power_kw = (
            current_step.base_load_power_forecast_kw
            + (current_step.hp_electric_power_forecast_kw * int(hp_on))
            - current_step.pv_available_power_forecast_kw
        )
        grid_import_kw = max(net_power_kw, 0.0)
        grid_export_kw = max(-net_power_kw, 0.0)
        return float(
            (
                (current_step.import_price_eur_kwh * grid_import_kw)
                - (current_step.export_price_eur_kwh * grid_export_kw)
            )
            * (interval_minutes / 60.0)
        )

    @staticmethod
    def _site_energy_cost_delta(
        *,
        current_step: MpcHorizonStep,
        hp_on: bool,
        interval_minutes: int,
    ) -> float:
        return float(
            SpaceHeatingMpcBacktestRunner._site_energy_cost(
                current_step=current_step,
                hp_on=hp_on,
                interval_minutes=interval_minutes,
            )
            - (
                SpaceHeatingMpcBacktestRunner._baseline_site_energy_cost(current_step)
                * (interval_minutes / 60.0)
            )
        )

    @staticmethod
    def _baseline_site_energy_cost(current_step: MpcHorizonStep) -> float:
        baseline_net_power_kw = (
            current_step.base_load_power_forecast_kw - current_step.pv_available_power_forecast_kw
        )
        baseline_import_kw = max(baseline_net_power_kw, 0.0)
        baseline_export_kw = max(-baseline_net_power_kw, 0.0)
        return float(
            (current_step.import_price_eur_kwh * baseline_import_kw)
            - (current_step.export_price_eur_kwh * baseline_export_kw)
        )

    @staticmethod
    def _summarize_step_results(
        step_results: list[MpcBacktestStepResult],
        *,
        interval_minutes: int,
        mode: str,
        q_heat_eff_active_threshold_kw: float = 0.1,
        infeasible_count: int = 0,
        slack_usage_count: int = 0,
        average_solver_runtime_seconds: float = 0.0,
    ) -> MpcBacktestSummary:
        simulated_days = max((len(step_results) * interval_minutes) / (24 * 60), 1e-9)
        comfort_violation_minutes = 0
        degree_minutes_below = 0.0
        degree_minutes_above = 0.0
        active_comfort_high_degree_minutes = 0.0
        passive_comfort_high_degree_minutes = 0.0
        runtime_minutes = 0
        energy_cost = 0.0
        starts = 0
        previous_hp_on: bool | None = None

        for step in step_results:
            hp_on = step.mpc_hp_on if mode == "mpc" else step.historical_hp_on
            q_heat_eff_kw = step.q_heat_eff_kw if mode == "mpc" else step.historical_q_heat_eff_kw
            room_temp = (
                step.simulated_next_room_temp_c
                if mode == "mpc"
                else step.historical_next_room_temp_c
            )
            energy_cost += (
                step.estimated_mpc_energy_cost_eur
                if mode == "mpc"
                else step.estimated_historical_energy_cost_eur
            )
            if hp_on:
                runtime_minutes += interval_minutes
            if previous_hp_on is not None and hp_on and not previous_hp_on:
                starts += 1
            previous_hp_on = hp_on
            if room_temp is None:
                continue
            below_comfort = max(step.temp_min_c - room_temp, 0.0)
            above_comfort = max(room_temp - step.temp_max_c, 0.0)
            if below_comfort > 0.0 or above_comfort > 0.0:
                comfort_violation_minutes += interval_minutes
            degree_minutes_below += below_comfort * interval_minutes
            degree_minutes_above += above_comfort * interval_minutes
            if above_comfort > 0.0:
                if hp_on or q_heat_eff_kw > q_heat_eff_active_threshold_kw:
                    active_comfort_high_degree_minutes += above_comfort * interval_minutes
                else:
                    passive_comfort_high_degree_minutes += above_comfort * interval_minutes

        return MpcBacktestSummary(
            comfort_violation_minutes=comfort_violation_minutes,
            degree_minutes_below_comfort=degree_minutes_below,
            degree_minutes_above_comfort=degree_minutes_above,
            active_comfort_high_degree_minutes=active_comfort_high_degree_minutes,
            passive_comfort_high_degree_minutes=passive_comfort_high_degree_minutes,
            starts_per_day=starts / simulated_days,
            runtime_minutes=runtime_minutes,
            estimated_energy_cost_eur=energy_cost,
            average_solver_runtime_seconds=average_solver_runtime_seconds if mode == "mpc" else 0.0,
            infeasible_count=infeasible_count if mode == "mpc" else 0,
            slack_usage_count=slack_usage_count if mode == "mpc" else 0,
        )

    @staticmethod
    def _pv_diagnostics(
        step_results: list[MpcBacktestStepResult],
        *,
        interval_minutes: int,
    ) -> MpcBacktestPvDiagnostics:
        dt_hours = interval_minutes / 60.0
        realized_pv_surplus_kwh = 0.0
        forecast_pv_surplus_kwh = 0.0
        mpc_hp_energy_kwh = 0.0
        mpc_hp_energy_during_realized_pv_surplus_kwh = 0.0
        mpc_hp_energy_during_forecast_pv_surplus_kwh = 0.0
        mpc_realized_pv_surplus_capture_kwh = 0.0
        preheat_budget_electric_kwh = 0.0
        used_preheat_budget_kwh = 0.0
        missed_surplus_with_headroom_kwh = 0.0
        run_durations_minutes: list[int] = []
        current_run_minutes = 0
        preheat_block_ids: set[int] = set()
        mpc_start_count = 0
        previous_hp_on = False

        for step in step_results:
            realized_pv_surplus_kwh += step.pv_surplus_realized_kw * dt_hours
            forecast_pv_surplus_kwh += step.pv_surplus_forecast_kw * dt_hours
            hp_energy_kwh = step.hp_electric_power_kw * float(int(step.mpc_hp_on)) * dt_hours
            mpc_hp_energy_kwh += hp_energy_kwh
            if step.preheat_block_id is not None:
                preheat_block_ids.add(step.preheat_block_id)
            preheat_budget_electric_kwh += step.preheat_budget_share_kwh
            if step.preheat_active:
                used_preheat_budget_kwh += step.preheat_charge_kwh
            mpc_hp_energy_during_realized_pv_surplus_kwh += (
                min(
                    step.hp_electric_power_kw * float(int(step.mpc_hp_on)),
                    step.pv_surplus_realized_kw,
                )
                * dt_hours
            )
            mpc_hp_energy_during_forecast_pv_surplus_kwh += (
                min(
                    step.hp_electric_power_kw * float(int(step.mpc_hp_on)),
                    step.pv_surplus_forecast_kw,
                )
                * dt_hours
            )
            mpc_realized_pv_surplus_capture_kwh += (
                min(
                    step.hp_electric_power_kw * float(int(step.mpc_hp_on)),
                    step.pv_surplus_realized_kw,
                )
                * dt_hours
            )
            if step.preheat_active:
                missed_surplus_with_headroom_kwh += max(
                    min(step.hp_electric_power_kw, step.pv_surplus_realized_kw) * dt_hours
                    - min(
                        step.hp_electric_power_kw * float(int(step.mpc_hp_on)),
                        step.pv_surplus_realized_kw,
                    )
                    * dt_hours,
                    0.0,
                )
            if step.mpc_hp_on:
                if not previous_hp_on:
                    mpc_start_count += 1
                current_run_minutes += interval_minutes
            elif current_run_minutes > 0:
                run_durations_minutes.append(current_run_minutes)
                current_run_minutes = 0
            previous_hp_on = step.mpc_hp_on
        if current_run_minutes > 0:
            run_durations_minutes.append(current_run_minutes)

        return MpcBacktestPvDiagnostics(
            realized_pv_surplus_kwh=realized_pv_surplus_kwh,
            forecast_pv_surplus_kwh=forecast_pv_surplus_kwh,
            mpc_hp_energy_kwh=mpc_hp_energy_kwh,
            mpc_hp_energy_during_realized_pv_surplus_kwh=(
                mpc_hp_energy_during_realized_pv_surplus_kwh
            ),
            mpc_hp_energy_during_forecast_pv_surplus_kwh=(
                mpc_hp_energy_during_forecast_pv_surplus_kwh
            ),
            mpc_realized_pv_surplus_capture_kwh=mpc_realized_pv_surplus_capture_kwh,
            mpc_realized_pv_surplus_capture_ratio=(
                mpc_realized_pv_surplus_capture_kwh / realized_pv_surplus_kwh
                if realized_pv_surplus_kwh > 0.0
                else 0.0
            ),
            mpc_forecast_pv_surplus_capture_ratio=(
                mpc_hp_energy_during_forecast_pv_surplus_kwh / forecast_pv_surplus_kwh
                if forecast_pv_surplus_kwh > 0.0
                else 0.0
            ),
            preheat_budget_electric_kwh=preheat_budget_electric_kwh,
            used_preheat_budget_kwh=used_preheat_budget_kwh,
            missed_surplus_with_headroom_kwh=missed_surplus_with_headroom_kwh,
            captured_realized_pv_kwh=mpc_realized_pv_surplus_capture_kwh,
            capture_ratio_realized=(
                mpc_realized_pv_surplus_capture_kwh / realized_pv_surplus_kwh
                if realized_pv_surplus_kwh > 0.0
                else 0.0
            ),
            average_run_duration_minutes=(
                sum(run_durations_minutes) / len(run_durations_minutes)
                if run_durations_minutes
                else 0.0
            ),
            short_run_count=sum(1 for duration in run_durations_minutes if duration < 30),
            preheat_block_count=len(preheat_block_ids),
            starts_per_preheat_block=(
                mpc_start_count / len(preheat_block_ids) if preheat_block_ids else 0.0
            ),
        )

    @staticmethod
    def _executed_preheat_budget_shortfall_cost(
        *,
        step_results: list[MpcBacktestStepResult],
        interval_minutes: int,
        objective_weights: MpcObjectiveWeights,
    ) -> float:
        dt_hours = interval_minutes / 60.0
        block_budget_kwh: dict[int, float] = {}
        block_charge_kwh: dict[int, float] = {}
        for step in step_results:
            if step.preheat_block_id is None:
                continue
            block_budget_kwh[step.preheat_block_id] = (
                block_budget_kwh.get(step.preheat_block_id, 0.0) + step.preheat_budget_share_kwh
            )
            block_charge_kwh[step.preheat_block_id] = (
                block_charge_kwh.get(step.preheat_block_id, 0.0)
                + min(
                    step.hp_electric_power_kw * float(int(step.mpc_hp_on)),
                    step.pv_surplus_forecast_kw,
                )
                * dt_hours
            )
        return sum(
            objective_weights.preheat_budget_shortfall
            * max(block_budget_kwh[block_id] - block_charge_kwh.get(block_id, 0.0), 0.0)
            for block_id in block_budget_kwh
        )
