from __future__ import annotations

from datetime import datetime

from home_optimizer.features.backtest.models import (
    MpcBacktestResult,
    MpcBacktestStepResult,
    MpcBacktestSummary,
)
from home_optimizer.features.mpc.controller_service import SpaceHeatingMpcControllerService
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcConstraints,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveBreakdown,
    MpcObjectiveWeights,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)


class SpaceHeatingMpcBacktestRunner:
    def __init__(
        self,
        *,
        controller: SpaceHeatingMpcControllerService | None = None,
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
        step_results: list[MpcBacktestStepResult] = []
        infeasible_count = 0
        total_solver_runtime_seconds = 0.0
        slack_usage_count = 0
        cumulative_objective_breakdown = MpcObjectiveBreakdown()
        historical_hp_on_by_timestamp = historical_hp_on_by_timestamp or {}
        historical_energy_cost_by_timestamp = historical_energy_cost_by_timestamp or {}

        for index in range(len(timeline) - 1):
            horizon = timeline[index : index + horizon_steps]
            if not horizon:
                break

            request = MpcControllerRequest(
                interval_minutes=interval_minutes,
                horizon=horizon,
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
            cumulative_objective_breakdown = self._add_objective_breakdowns(
                cumulative_objective_breakdown,
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
            current_step = timeline[index]
            q_heat_eff_kw = self._apply_heat_actuator(
                control_model=control_model,
                previous_q_heat_eff_kw=current_state.q_heat_eff_kw,
                commanded_hp_on=applied_hp_on,
                current_step=current_step,
            )
            predicted_next_temp, next_state = self._predict_next_state(
                control_model=control_model,
                current_state=current_state,
                current_step=current_step,
                heating_effect_kw=q_heat_eff_kw,
            )
            total_solver_runtime_seconds += solve_time_seconds or 0.0

            historical_hp_on = historical_hp_on_by_timestamp.get(
                current_step.timestamp_utc,
                current_step.effective_heating_kw_forecast > 0.0,
            )

            step_results.append(
                MpcBacktestStepResult(
                    timestamp_utc=current_step.timestamp_utc,
                    mpc_hp_on=applied_hp_on,
                    historical_hp_on=historical_hp_on,
                    start=start,
                    stop=stop,
                    q_heat_eff_kw=q_heat_eff_kw,
                    predicted_next_room_temp_c=predicted_next_temp,
                    simulated_next_room_temp_c=predicted_next_temp,
                    historical_next_room_temp_c=next_step.realized_room_temp_c,
                    temp_min_c=next_step.temp_min_c,
                    temp_max_c=next_step.temp_max_c,
                    slack_low_c=slack_low_c,
                    slack_high_c=slack_high_c,
                    price_eur_kwh=current_step.import_price_eur_kwh,
                    estimated_mpc_energy_cost_eur=self._site_energy_cost(
                        current_step=current_step,
                        hp_on=applied_hp_on,
                        interval_minutes=interval_minutes,
                    ),
                    estimated_historical_energy_cost_eur=self._site_energy_cost(
                        current_step=current_step,
                        hp_on=historical_hp_on,
                        interval_minutes=interval_minutes,
                        override_cost=historical_energy_cost_by_timestamp.get(
                            current_step.timestamp_utc
                        ),
                    ),
                    solve_time_seconds=solve_time_seconds,
                    feasible=plan.feasible,
                )
            )

            current_state = self._advance_state(
                next_state=next_state.model_copy(update={"q_heat_eff_kw": q_heat_eff_kw}),
                hp_on=applied_hp_on,
            )

        average_solver_runtime = total_solver_runtime_seconds / len(step_results) if step_results else 0.0
        return MpcBacktestResult(
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
                infeasible_count=infeasible_count,
                slack_usage_count=slack_usage_count,
                average_solver_runtime_seconds=average_solver_runtime,
            ),
            historical_summary=self._summarize_step_results(
                step_results,
                interval_minutes=interval_minutes,
                mode="historical",
            ),
            mpc_objective_breakdown=cumulative_objective_breakdown,
            total_solver_runtime_seconds=total_solver_runtime_seconds,
        )

    @staticmethod
    def _add_objective_breakdowns(
        left: MpcObjectiveBreakdown,
        right: MpcObjectiveBreakdown,
    ) -> MpcObjectiveBreakdown:
        return MpcObjectiveBreakdown(
            comfort_low=left.comfort_low + right.comfort_low,
            comfort_high=left.comfort_high + right.comfort_high,
            temperature_tracking=left.temperature_tracking + right.temperature_tracking,
            terminal=left.terminal + right.terminal,
            start=left.start + right.start,
            runtime=left.runtime + right.runtime,
            energy=left.energy + right.energy,
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
    def _summarize_step_results(
        step_results: list[MpcBacktestStepResult],
        *,
        interval_minutes: int,
        mode: str,
        infeasible_count: int = 0,
        slack_usage_count: int = 0,
        average_solver_runtime_seconds: float = 0.0,
    ) -> MpcBacktestSummary:
        simulated_days = max((len(step_results) * interval_minutes) / (24 * 60), 1e-9)
        comfort_violation_minutes = 0
        degree_minutes_below = 0.0
        degree_minutes_above = 0.0
        runtime_minutes = 0
        energy_cost = 0.0
        starts = 0
        previous_hp_on: bool | None = None

        for step in step_results:
            hp_on = step.mpc_hp_on if mode == "mpc" else step.historical_hp_on
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

        return MpcBacktestSummary(
            comfort_violation_minutes=comfort_violation_minutes,
            degree_minutes_below_comfort=degree_minutes_below,
            degree_minutes_above_comfort=degree_minutes_above,
            starts_per_day=starts / simulated_days,
            runtime_minutes=runtime_minutes,
            estimated_energy_cost_eur=energy_cost,
            average_solver_runtime_seconds=average_solver_runtime_seconds if mode == "mpc" else 0.0,
            infeasible_count=infeasible_count if mode == "mpc" else 0,
            slack_usage_count=slack_usage_count if mode == "mpc" else 0,
        )
