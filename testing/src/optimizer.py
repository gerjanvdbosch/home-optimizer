import cvxpy as cp
import pandas as pd
import numpy as np
import logging

from datetime import datetime

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
        outside_temp: float = 15,
        volume_liter: float = 200,
    ):
        """
        Simuleert het opwarmproces.
        TODO: train met model voor SWW en verwarmen
        """
        if t_start_temp >= t_target:
            return np.array([])

        # Elektrisch vermogen (kW)
        # Gemiddeld verbruik SW75 tijdens SWW run is vaak rond de 2.2 - 2.8 kW
        p_rated = 2.5
        p_max_limit = 3.2
        p_temp_coeff = 0.015
        ramp_factors = [0.8, 0.9]

        current_water_temp = t_start_temp
        actual_profile_elec = []

        step_counter = 0
        max_steps = int(12 / self.timestep_hours)

        while current_water_temp < t_target and step_counter < max_steps:
            # 1. Elektrisch profiel
            p_target = p_rated + (current_water_temp - 40) * p_temp_coeff
            ramp_factor = ramp_factors[step_counter] if step_counter < len(ramp_factors) else 1.0
            p_elec = p_target * ramp_factor
            p_elec = max(0.5, min(p_max_limit, p_elec))

            lift = current_water_temp - outside_temp
            estimated_cop = 5.5 - (0.10 * lift)
            current_cop = np.clip(estimated_cop, 1.5, 4.5)

            # 3. Thermisch & Temperatuur
            p_thermal = p_elec * current_cop

            # Als p_thermal te hoog wordt, knijpen we hem af (saturation)
            if p_thermal > 8.0:
                 p_thermal = 8.0

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
            return 'WAIT', "Boiler al op temperatuur", 0.0, 0.0

        duration_steps = len(power_profile)

        # Filter op toekomst
        future = df[df["timestamp"] >= current_time].copy().reset_index(drop=True)
        horizon_steps = min(len(future), int(24 / self.timestep_hours))
        future = future.iloc[:horizon_steps]

        if len(future) < duration_steps:
            logger.warning(f"Niet genoeg data: {len(future)}")
            return 'WAIT', "Niet genoeg data in horizon", 0.0, 0.0

        P_solar = future["power_corrected"].values
        outside_temp = future["temp"].values
        P_load = future["load_corrected"].values
        T = len(P_solar)

        num_possible_starts = T - duration_steps + 1
        if num_possible_starts <= 0:
            return 'WAIT', "Niet genoeg tijd voor verwarmen", 0.0, 0.0

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
            return 'WAIT', "Optimalisatie mislukt", 0.0, 0.0

        best_start_idx = int(np.argmax(start_flags.value))
        planned_start = future.iloc[best_start_idx]["timestamp"]

        best_end_idx = best_start_idx + duration_steps
        slice_solar = P_solar[best_start_idx:best_end_idx]
        slice_dhw = power_profile
        slice_load = P_load[best_start_idx:best_end_idx] # Ook load slicen

        # Hoeveel van de boiler energie komt direct uit zon?
        # Formule: min(Solar, Boiler + Huis) - min(Solar, Huis)
        # Of simpeler benaderd voor context:
        # We kijken naar het totaalverbruik in dat raamwerk
        total_consumption = slice_dhw + slice_load
        direct_solar_usage = np.sum(np.minimum(slice_solar, total_consumption))
        # Maar we willen eigenlijk weten hoeveel *extra* solar de boiler pakt:
        base_solar_usage = np.sum(np.minimum(slice_solar, slice_load))
        boiler_solar_usage = direct_solar_usage - base_solar_usage

        # Als alternatief (simpeler): hoeveel solar dekt de som
        total_covered = np.sum(np.minimum(slice_solar, slice_dhw + slice_load))

        minutes_to_start = (planned_start - current_time).total_seconds() / 60
        start_local = planned_start.tz_convert(datetime.now().astimezone().tzinfo)

        status = 'START' if minutes_to_start <= 5 else 'WAIT'
        reason = "Starttijd bereikt" if status == 'START' else f"Start gepland om {start_local:%H:%M}"
        # Load van NU voor logging (eerste waarde van de vector)
        current_load_val = P_load[0] if len(P_load) > 0 else 0.05
        solar_usage_kwh = round(boiler_solar_usage * self.timestep_hours, 2)

        return status, reason, solar_usage_kwh, current_load_val
