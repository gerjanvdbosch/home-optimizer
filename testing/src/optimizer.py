import cvxpy as cp
import pandas as pd
import numpy as np
import logging

from datetime import datetime
from context import SolarStatus, SolarContext

logger = logging.getLogger(__name__)

"""
- Optimizer voor warmtepompboiler voor minimalisatie grid import en maximalisatie zonnegebruik
-- Gebaseerd op CVXPY is noodzakelijk
-- Dynamische simulatie van warmtepomp gedrag
-- Optimalisatie van starttijdstip gebaseerd op hoogste zonneproductie
-- Fallback verwarmen indien geen zon beschikbaar op moment met hoogste buiten temperatuur
-- Water moet warm zijn voor 17:00
-- 1 run per dag
"""
class Optimizer:
    def __init__(self, pv_max_kw: float):
        self.pv_max_kw = pv_max_kw
        self.timestep_hours = 0.25  # 15 min

    def calculate_profile(
        self,
        t_start_temp: float,
        t_target: float,
        volume_liter: float = 200,
        outside_temp: float = 15,
    ):
        """
        Simuleert het opwarmproces dynamisch op basis van temperaturen.
        """
        if t_start_temp >= t_target:
            return np.array([])

        p_rated = 2.7
        p_max_limit = 3.2
        p_temp_coeff = 0.015
        ramp_factors = [0.5, 0.8]

        current_water_temp = t_start_temp
        actual_profile_elec = []

        step_counter = 0
        max_steps = int(12 / self.timestep_hours)

        while current_water_temp < t_target and step_counter < max_steps:
            p_target = p_rated + (current_water_temp - 40) * p_temp_coeff

            ramp_factor = ramp_factors[step_counter] if step_counter < len(ramp_factors) else 1.0
            p_elec = p_target * ramp_factor
            p_elec = max(0.5, min(p_max_limit, p_elec))

            cop_base = 3.5
            cop_corr_air = (outside_temp - 15) * 0.06
            cop_corr_water = (current_water_temp - 40) * -0.04
            current_cop = np.clip(cop_base + cop_corr_air + cop_corr_water, 1.5, 5.0)

            p_thermal = p_elec * current_cop
            energy_thermal_kwh = p_thermal * self.timestep_hours

            delta_t = (energy_thermal_kwh * 3600) / (volume_liter * 4.18)
            current_water_temp += delta_t

            actual_profile_elec.append(p_elec)
            step_counter += 1

        return np.array(actual_profile_elec)

    def optimize(
        self, df: pd.DataFrame, current_time: pd.Timestamp, power_profile: np.ndarray
    ):
        if len(power_profile) == 0:
            return SolarStatus.WAIT, None

        duration_steps = len(power_profile)

        # Filter op toekomst
        future = df[df["timestamp"] >= current_time].copy().reset_index(drop=True)
        horizon_steps = min(len(future), int(24 / self.timestep_hours))
        future = future.iloc[:horizon_steps]

        if len(future) < duration_steps:
            logger.warning("Niet genoeg data (horizon korter dan benodigde duur).")
            return SolarStatus.WAIT, None

        P_solar = future["power_corrected"].values
        outside_temp = future["outside_temp"].values
        #P_load = future["projected_load"].values
        P_load = 0.25
        T = len(P_solar)

        num_possible_starts = T - duration_steps + 1
        if num_possible_starts <= 0:
            return SolarStatus.WAIT, None

        start_flags = cp.Variable(num_possible_starts, boolean=True)
        dhw_power_vector = cp.Variable(T)

        constraints = [cp.sum(start_flags) == 1]

        profile_expr = 0
        for k in range(num_possible_starts):
            vec = np.zeros(T)
            vec[k : k + duration_steps] = power_profile
            profile_expr += start_flags[k] * vec

        constraints.append(dhw_power_vector == profile_expr)

        # --- OBJECTIVE ---
        net_load = (P_load + dhw_power_vector) - P_solar
        grid_import = cp.pos(net_load)

        # secundaire term: warmere buitentemp = lagere straf
        temp_norm = (outside_temp - np.mean(outside_temp)) / (np.std(outside_temp) + 1e-6)
        temp_penalty = cp.sum(cp.multiply(dhw_power_vector, -temp_norm))

        lambda_cop = 0.05  # klein: grid blijft dominant

        objective = cp.Minimize(
            cp.sum(grid_import) + lambda_cop * temp_penalty
        )

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve()
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            return SolarStatus.WAIT, None

        best_start_idx = int(np.argmax(start_flags.value))
        planned_start = future.iloc[best_start_idx]["timestamp"]

        best_end_idx = best_start_idx + duration_steps
        slice_solar = P_solar[best_start_idx:best_end_idx]
        slice_dhw = power_profile

        direct_solar_usage = np.sum(np.minimum(slice_solar, slice_dhw + P_load))

        minutes_to_start = (planned_start - current_time).total_seconds() / 60
        start_local = planned_start.tz_convert(datetime.now().astimezone().tzinfo)

        status = SolarStatus.START if minutes_to_start <= 5 else SolarStatus.WAIT
        reason = "Starttijd bereikt." if status == SolarStatus.START else f"Start gepland om {start_local:%H:%M}"

        return status, SolarContext(
            actual_pv=0,
            energy_now=0,
            energy_best=round(direct_solar_usage * self.timestep_hours, 2),
            opportunity_cost=0,
            confidence=1.0,
            action=status,
            reason=reason,
            planned_start=start_local,
            load_now=P_load,
        )
