from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thermal_model import ThermalModel
from .types import ForecastHorizon, MPCParameters

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only when cvxpy is unavailable
    cp = None


@dataclass(frozen=True, slots=True)
class MPCSolution:
    control_sequence_kw: np.ndarray
    predicted_states_c: np.ndarray
    objective_value: float
    solver_status: str
    used_fallback: bool = False

    @property
    def first_control_kw(self) -> float:
        if self.control_sequence_kw.size == 0:
            return 0.0
        return float(self.control_sequence_kw[0])


class UFHMPCController:
    def __init__(
        self,
        model: ThermalModel,
        parameters: MPCParameters,
        solver: str = "OSQP",
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.solver = solver

    def solve(
        self,
        initial_state_c: np.ndarray,
        forecast: ForecastHorizon,
        previous_power_kw: float = 0.0,
    ) -> MPCSolution:
        initial_state_c = np.asarray(initial_state_c, dtype=float)
        if initial_state_c.shape != (2,):
            raise ValueError("initial_state_c must be a 2-vector ordered as [T_r, T_b].")
        if forecast.horizon_steps != self.parameters.horizon_steps:
            raise ValueError("Forecast horizon length must match MPCParameters.horizon_steps.")

        if cp is not None:
            try:
                return self._solve_convex(
                    initial_state_c=initial_state_c,
                    forecast=forecast,
                    previous_power_kw=previous_power_kw,
                )
            except cp.SolverError:
                pass

        return self._solve_greedy(
            initial_state_c=initial_state_c,
            forecast=forecast,
            previous_power_kw=previous_power_kw,
        )

    def _solve_convex(
        self,
        initial_state_c: np.ndarray,
        forecast: ForecastHorizon,
        previous_power_kw: float,
    ) -> MPCSolution:
        assert cp is not None

        A, B, E = self.model.state_matrices()
        B_vec = B[:, 0]
        disturbances = forecast.disturbance_matrix(self.model.parameters)
        settings = self.parameters
        dt = self.model.parameters.dt_hours
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh
        horizon = settings.horizon_steps

        state = cp.Variable((2, horizon + 1))
        control = cp.Variable(horizon)

        constraints: list[cp.Constraint] = [state[:, 0] == initial_state_c]
        objective_terms = []

        for step in range(horizon):
            constraints.append(
                state[:, step + 1]
                == A @ state[:, step] + B_vec * control[step] + E @ disturbances[step]
            )
            constraints.extend(
                [
                    control[step] >= 0.0,
                    control[step] <= settings.P_max,
                    state[0, step + 1] >= settings.T_min,
                    state[0, step + 1] <= settings.T_max,
                ]
            )
            if step == 0:
                constraints.append(
                    cp.abs(control[step] - previous_power_kw) <= settings.delta_P_max
                )
            else:
                constraints.append(
                    cp.abs(control[step] - control[step - 1]) <= settings.delta_P_max
                )

            objective_terms.append(
                settings.Q_c * cp.square(state[0, step] - refs[step])
                + prices[step] * control[step] * dt
                + settings.R_c * cp.square(control[step])
            )

        objective = cp.Minimize(
            cp.sum(objective_terms) + settings.Q_N * cp.square(state[0, horizon] - refs[horizon])
        )
        problem = cp.Problem(objective, constraints)
        solve_kwargs: dict[str, object] = {
            "solver": self.solver,
            "warm_start": True,
            "verbose": False,
        }
        if self.solver.upper() == "OSQP":
            solve_kwargs.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "polishing": True})
        problem.solve(**solve_kwargs)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"MPC solve failed with status {problem.status}.")
        if control.value is None or state.value is None:
            raise ValueError("MPC solver did not return a solution.")

        return MPCSolution(
            control_sequence_kw=np.asarray(control.value, dtype=float).reshape(horizon),
            predicted_states_c=np.asarray(state.value, dtype=float).T,
            objective_value=float(problem.value),
            solver_status=str(problem.status),
            used_fallback=False,
        )

    def _solve_greedy(
        self,
        initial_state_c: np.ndarray,
        forecast: ForecastHorizon,
        previous_power_kw: float,
    ) -> MPCSolution:
        A, B, E = self.model.state_matrices()
        B_vec = B[:, 0]
        disturbances = forecast.disturbance_matrix(self.model.parameters)
        settings = self.parameters
        dt = self.model.parameters.dt_hours
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh

        state = initial_state_c.copy()
        states = [state.copy()]
        controls = []
        previous_control = previous_power_kw

        for step in range(settings.horizon_steps):
            lower = max(0.0, previous_control - settings.delta_P_max)
            upper = min(settings.P_max, previous_control + settings.delta_P_max)
            if np.isclose(lower, upper):
                candidates = np.array([lower], dtype=float)
            else:
                resolution = max(settings.delta_P_max / 5.0, 0.25)
                candidate_count = int(np.ceil((upper - lower) / resolution)) + 1
                candidate_count = min(max(candidate_count, 5), 25)
                candidates = np.linspace(lower, upper, candidate_count, dtype=float)

            best_control = candidates[0]
            best_next_state = None
            best_score = np.inf

            for candidate in candidates:
                next_state = A @ state + B_vec * candidate + E @ disturbances[step]
                comfort_violation = (
                    max(settings.T_min - next_state[0], 0.0) ** 2
                    + max(next_state[0] - settings.T_max, 0.0) ** 2
                )
                score = (
                    settings.Q_c * (state[0] - refs[step]) ** 2
                    + prices[step] * candidate * dt
                    + settings.R_c * candidate**2
                    + 1_000_000.0 * comfort_violation
                    + 5.0 * settings.Q_N * max(refs[step + 1] - next_state[0], 0.0) ** 2
                )
                if step == settings.horizon_steps - 1:
                    score += settings.Q_N * (next_state[0] - refs[-1]) ** 2

                if score < best_score:
                    best_score = score
                    best_control = float(candidate)
                    best_next_state = next_state

            assert best_next_state is not None
            controls.append(best_control)
            state = best_next_state
            states.append(state.copy())
            previous_control = best_control

        controls_array = np.asarray(controls, dtype=float)
        states_array = np.asarray(states, dtype=float)
        objective = self._objective_value(states_array, controls_array, forecast)
        return MPCSolution(
            control_sequence_kw=controls_array,
            predicted_states_c=states_array,
            objective_value=objective,
            solver_status="fallback-greedy",
            used_fallback=True,
        )

    def _objective_value(
        self,
        predicted_states_c: np.ndarray,
        control_sequence_kw: np.ndarray,
        forecast: ForecastHorizon,
    ) -> float:
        settings = self.parameters
        dt = self.model.parameters.dt_hours
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh

        stage_cost = 0.0
        for step, control_kw in enumerate(control_sequence_kw):
            stage_cost += (
                settings.Q_c * (predicted_states_c[step, 0] - refs[step]) ** 2
                + prices[step] * control_kw * dt
                + settings.R_c * control_kw**2
            )
        terminal_cost = settings.Q_N * (predicted_states_c[-1, 0] - refs[-1]) ** 2
        return float(stage_cost + terminal_cost)
