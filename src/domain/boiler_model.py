from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class BoilerThermalModel:
    """Fysisch 2-zone thermisch boilermodel.

    Dient als Single Source of Truth voor zowel NumPy-simulaties
    (predict/backtest) als Pyomo MILP-formuleringen (MPC).
    """

    c_top: float = 0.1163  # Warmtecapaciteit bovenste zone (kWh/°C)
    c_bottom: float = 0.1163  # Warmtecapaciteit onderste zone (kWh/°C)
    ua_top: float = 0.0014  # Warmteverliescoëfficiënt boven (kW/°C)
    ua_bottom: float = 0.0014  # Warmteverliescoëfficiënt onder (kW/°C)
    k_idle: float = 0.0003  # Interne geleiding in rust (kW/°C)
    k_mix: float = 0.0400  # Geforceerde menging tijdens bedrijf (kW/°C)
    f_top: float = 0.50  # Fractie warmtepompvermogen naar bovenlaag
    typical_q_hp_kw: float = 5.5  # Typisch geleverd thermisch vermogen (kW)
    dt_hours: float = 0.25  # Tijdstap in uren (15 min = 0.25)

    def step(
        self,
        t_top: float,
        t_bottom: float,
        t_ambient: float,
        q_hp: float,
    ) -> tuple[float, float]:
        """Eén discrete Euler-integratiestap (NumPy / pure Python)."""
        is_active = 1.0 if q_hp > 0.5 else 0.0
        k_eff = self.k_idle + self.k_mix * is_active

        # Toplaag
        dq_top = (
            self.f_top * q_hp
            + self.ua_top * (t_ambient - t_top)
            + k_eff * (t_bottom - t_top)
        )
        t_top_next = t_top + (dq_top * self.dt_hours) / self.c_top

        # Onderlaag
        dq_bottom = (
            (1.0 - self.f_top) * q_hp
            + self.ua_bottom * (t_ambient - t_bottom)
            - k_eff * (t_bottom - t_top)
        )
        t_bottom_next = t_bottom + (dq_bottom * self.dt_hours) / self.c_bottom

        return t_top_next, t_bottom_next

    def simulate(
        self,
        initial_top: float,
        initial_bottom: float,
        ambient_temps: np.ndarray,
        q_hp_profile: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Voorwaartse simulatie over een gegeven tijdshorizon."""
        n = len(ambient_temps)
        sim_top = np.empty(n, dtype=float)
        sim_bot = np.empty(n, dtype=float)

        sim_top[0] = initial_top
        sim_bot[0] = initial_bottom

        for t in range(n - 1):
            sim_top[t + 1], sim_bot[t + 1] = self.step(
                t_top=sim_top[t],
                t_bottom=sim_bot[t],
                t_ambient=ambient_temps[t],
                q_hp=q_hp_profile[t],
            )

        return sim_top, sim_bot

    def apply_pyomo_constraints(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        t_min: float = 10.0,
        t_max: float = 75.0,
    ) -> None:
        """Bouwt de exacte McCormick & ODE dynamica in het Pyomo model."""
        dt = self.dt_hours
        diff_max = t_max - t_min
        diff_min = t_min - t_max
        t_amb = float(data.ambient_temperature)

        # 1. Variabelen
        model.T_top = pyo.Var(model.T, bounds=(t_min, t_max))
        model.T_bottom = pyo.Var(model.T, bounds=(t_min, t_max))
        model.delta_mix = pyo.Var(model.T, bounds=(diff_min, diff_max))
        model.q_curtail = pyo.Var(model.T, bounds=(0.0, self.typical_q_hp_kw))

        # 2. Startcondities
        model.init_top = pyo.Constraint(expr=model.T_top[0] == data.current_temp_top)
        model.init_bottom = pyo.Constraint(
            expr=model.T_bottom[0] == data.current_temp_bottom
        )

        # 3. Inverter curtailment limit
        model.curtail_limit = pyo.Constraint(
            model.T,
            rule=lambda m, t: m.q_curtail[t] <= self.typical_q_hp_kw * m.boiler_on[t],
        )

        # 4. McCormick relaxatie voor niet-lineaire menging (k_mix * boiler_on * delta_T)
        model.mix_ub_on = pyo.Constraint(
            model.T, rule=lambda m, t: m.delta_mix[t] <= diff_max * m.boiler_on[t]
        )
        model.mix_lb_on = pyo.Constraint(
            model.T, rule=lambda m, t: m.delta_mix[t] >= diff_min * m.boiler_on[t]
        )
        model.mix_ub_diff = pyo.Constraint(
            model.T,
            rule=lambda m, t: (
                m.delta_mix[t]
                <= (m.T_bottom[t] - m.T_top[t]) + diff_max * (1 - m.boiler_on[t])
            ),
        )
        model.mix_lb_diff = pyo.Constraint(
            model.T,
            rule=lambda m, t: (
                m.delta_mix[t]
                >= (m.T_bottom[t] - m.T_top[t]) + diff_min * (1 - m.boiler_on[t])
            ),
        )

        # 5. Fysische toestandsovergangen
        def top_dynamics_rule(m, t):
            if t == len(m.T) - 1:
                return pyo.Constraint.Skip
            q_hp = (self.typical_q_hp_kw * m.boiler_on[t]) - m.q_curtail[t]
            q_in = self.f_top * q_hp
            q_loss = self.ua_top * (t_amb - m.T_top[t])
            q_cond = self.k_idle * (m.T_bottom[t] - m.T_top[t])
            q_mix = self.k_mix * m.delta_mix[t]
            dT = (q_in + q_loss + q_cond + q_mix) * dt / self.c_top
            return m.T_top[t + 1] == m.T_top[t] + dT

        model.top_dynamics = pyo.Constraint(model.T, rule=top_dynamics_rule)

        def bottom_dynamics_rule(m, t):
            if t == len(m.T) - 1:
                return pyo.Constraint.Skip
            q_hp = (self.typical_q_hp_kw * m.boiler_on[t]) - m.q_curtail[t]
            q_in = (1.0 - self.f_top) * q_hp
            q_loss = self.ua_bottom * (t_amb - m.T_bottom[t])
            q_cond = -self.k_idle * (m.T_bottom[t] - m.T_top[t])
            q_mix = -self.k_mix * m.delta_mix[t]
            dT = (q_in + q_loss + q_cond + q_mix) * dt / self.c_bottom
            return m.T_bottom[t + 1] == m.T_bottom[t] + dT

        model.bottom_dynamics = pyo.Constraint(model.T, rule=bottom_dynamics_rule)


@dataclass(frozen=True)
class MPCConfig:
    step_hours: float = Field(default=0.25, gt=0)
    boiler_power: float = Field(default=2.0, gt=0)
    boiler_steps: int = Field(default=4, gt=0)
    max_starts: int = 1


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_kw: list[float]
    ambient_temperature: float
    current_temp_top: float
    current_temp_bottom: float
    boiler_on: bool
    thermal_model: BoilerThermalModel


@dataclass(frozen=True)
class MPCResult(BaseModel):
    schedule: tuple[int, ...]
    temperatures_top: tuple[float, ...]  # Gepland verloop bovenkant (°C)
    temperatures_bottom: tuple[float, ...]  # Gepland verloop onderkant (°C)
    objective_value: float
    solver_status: str
    termination_condition: str
