from __future__ import annotations

from home_optimizer.features.mpc.controller_service import SpaceHeatingMpcControllerService
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcBacktestResult,
    MpcBacktestStepResult,
    MpcConstraints,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveWeights,
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
        control_model: LinearThermalControlModel,
        timeline: list[MpcHorizonStep],
        initial_state: MpcInitialState,
        interval_minutes: int,
        horizon_steps: int,
        constraints: MpcConstraints | None = None,
        objective_weights: MpcObjectiveWeights | None = None,
        max_solver_seconds: float | None = None,
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
        total_starts = 0
        total_runtime_minutes = 0
        total_energy_cost = 0.0
        comfort_violation_minutes = 0
        degree_minutes_below = 0.0
        degree_minutes_above = 0.0
        slack_usage_count = 0

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
                total_energy_cost += first_step.estimated_energy_cost_eur
                total_starts += int(first_step.start)
                total_runtime_minutes += interval_minutes if first_step.hp_on else 0
                if first_step.slack_low_c > 0.0 or first_step.slack_high_c > 0.0:
                    slack_usage_count += 1

            next_step = timeline[index + 1]
            current_step = timeline[index]
            heating_effect_kw = current_step.effective_heating_kw_forecast * int(applied_hp_on)
            predicted_next_temp = control_model.predict_next_temperature(
                room_temp_c=current_state.room_temp_c,
                outdoor_temp_c=current_step.outdoor_temp_c,
                solar_gain_kw=current_step.solar_gain_kw,
                heating_effect_kw=heating_effect_kw,
                occupied=current_step.occupied,
            )
            realized_next_temp = (
                next_step.realized_room_temp_c
                if next_step.realized_room_temp_c is not None
                else predicted_next_temp
            )

            below_comfort = max(next_step.temp_min_c - realized_next_temp, 0.0)
            above_comfort = max(realized_next_temp - next_step.temp_max_c, 0.0)
            if below_comfort > 0.0 or above_comfort > 0.0:
                comfort_violation_minutes += interval_minutes
            degree_minutes_below += below_comfort * interval_minutes
            degree_minutes_above += above_comfort * interval_minutes
            total_solver_runtime_seconds += solve_time_seconds or 0.0

            step_results.append(
                MpcBacktestStepResult(
                    timestamp_utc=current_step.timestamp_utc,
                    hp_on=applied_hp_on,
                    start=start,
                    stop=stop,
                    predicted_next_room_temp_c=predicted_next_temp,
                    realized_next_room_temp_c=realized_next_temp,
                    temp_min_c=next_step.temp_min_c,
                    temp_max_c=next_step.temp_max_c,
                    slack_low_c=slack_low_c,
                    slack_high_c=slack_high_c,
                    estimated_energy_cost_eur=current_step.price_eur_kwh
                    * heating_effect_kw
                    * (interval_minutes / 60.0),
                    solve_time_seconds=solve_time_seconds,
                    feasible=plan.feasible,
                )
            )

            current_state = self._advance_state(
                room_temp_c=realized_next_temp,
                hp_on=applied_hp_on,
                previous_state=current_state,
            )

        simulated_days = max((len(step_results) * interval_minutes) / (24 * 60), 1e-9)
        average_solver_runtime = (
            total_solver_runtime_seconds / len(step_results) if step_results else 0.0
        )
        return MpcBacktestResult(
            step_results=step_results,
            comfort_violation_minutes=comfort_violation_minutes,
            degree_minutes_below_comfort=degree_minutes_below,
            degree_minutes_above_comfort=degree_minutes_above,
            starts_per_day=total_starts / simulated_days,
            runtime_minutes=total_runtime_minutes,
            estimated_energy_cost_eur=total_energy_cost,
            total_solver_runtime_seconds=total_solver_runtime_seconds,
            average_solver_runtime_seconds=average_solver_runtime,
            infeasible_count=infeasible_count,
            slack_usage_count=slack_usage_count,
        )

    @staticmethod
    def _advance_state(
        *,
        room_temp_c: float,
        hp_on: bool,
        previous_state: MpcInitialState,
    ) -> MpcInitialState:
        if hp_on:
            on_steps = previous_state.on_steps + 1 if previous_state.hp_on else 1
            return MpcInitialState(
                room_temp_c=room_temp_c,
                hp_on=True,
                on_steps=on_steps,
                off_steps=0,
            )
        off_steps = previous_state.off_steps + 1 if not previous_state.hp_on else 1
        return MpcInitialState(
            room_temp_c=room_temp_c,
            hp_on=False,
            on_steps=0,
            off_steps=off_steps,
        )
