import cvxpy as cp
import pandas as pd
import numpy as np
import logging

from context import SolarStatus, SolarContext

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, pv_max_kw: float):
        self.pv_max_kw = pv_max_kw
        self.timestep_hours = 0.25  # Kwartierwaarden

    def calculate_profile(
        self,
        t_start_temp: float,
        t_target: float,
        volume_liter: float = 200,
        outside_temp: float = 15,
    ):
        """
        Simuleert het opwarmproces van een warmtepompboiler stap voor stap.
        Houdt rekening met COP-verlies bij koud weer en heter water.
        """
        if t_start_temp >= t_target:
            return np.array([])

        # 1. Configuratie van de Warmtepomp (Aanpasbaar aan jouw model)
        # Elektrisch vermogen curve (optoeren compressor)
        # Dit is wat hij uit het stopcontact trekt.
        ramp_up_curve = [1.5, 1.7, 2.0, 2.3, 2.5]
        max_power_elec = 2.7

        # 2. Start Simulatie
        current_water_temp = t_start_temp
        actual_profile_elec = []

        step_counter = 0

        # We stoppen als het water warm is OF als we absurd lang bezig zijn (beveiliging > 12 uur)
        max_steps = int(12 / self.timestep_hours)

        while current_water_temp < t_target and step_counter < max_steps:

            # A. Bepaal elektrisch vermogen voor dit kwartier
            if step_counter < len(ramp_up_curve):
                p_elec = ramp_up_curve[step_counter]
            else:
                p_elec = max_power_elec

            # B. Schat de COP (Coefficient of Performance)
            # Dit is de cruciale stap voor nauwkeurigheid!
            # Basisregel: COP zakt als buitenlucht koud is, en als water heet is.
            # Voorbeeld formule voor een moderne WP boiler (R134a/R290):
            # COP ~ 3.5 bij (Air=15, Water=40).
            # -0.06 per graad lagere luchttemp
            # -0.04 per graad hogere watertemp

            cop_base = 3.5
            cop_corr_air = (outside_temp - 15) * 0.06  # Kouder buiten = lagere COP
            cop_corr_water = (
                current_water_temp - 40
            ) * -0.04  # Heter water = lagere COP

            current_cop = cop_base + cop_corr_air + cop_corr_water

            # Begrens de COP op realistische waarden (min 1.5, max 5.0)
            current_cop = max(1.5, min(5.0, current_cop))

            # C. Bereken Thermisch Vermogen (Wat gaat er het vat in?)
            p_thermal = p_elec * current_cop  # kW

            # D. Bereken temperatuurstijging in dit kwartier (0.25 uur)
            # Formule: Q (kWh) = m (kg) * c * deltaT / 3600
            # Dus: deltaT = (kWh_thermisch * 3600) / (m * c)
            # Soortelijke warmte water (c) is ong 4.18 kJ/kg.K

            energy_thermal_kwh = p_thermal * self.timestep_hours

            delta_t = (energy_thermal_kwh * 3600) / (volume_liter * 4.18)

            # Update de simulatie
            current_water_temp += delta_t
            actual_profile_elec.append(p_elec)
            step_counter += 1

        return np.array(actual_profile_elec)

    def optimize(
        self, df: pd.DataFrame, current_time: pd.Timestamp, power_profile: np.ndarray
    ):
        """
        Vindt het beste startmoment gegeven een specifiek vermogensprofiel.
        """
        if len(power_profile) == 0:
            return SolarStatus.WAIT, SolarContext(
                reason="Water is already hot",
                energy_best=0,
                action=SolarStatus.WAIT,
                actual_pv=0,
                energy_now=0,
                opportunity_cost=0,
                confidence=1.0,
                load_now=0,
            )

        duration_steps = len(power_profile)

        # Filter op toekomst
        future = df[df["timestamp"] >= current_time].copy().reset_index(drop=True)
        horizon_steps = min(len(future), int(24 / self.timestep_hours))
        future = future.iloc[:horizon_steps]

        if len(future) < duration_steps:
            logger.warning("Niet genoeg data (horizon korter dan benodigde duur).")
            return SolarStatus.WAIT, None

        P_solar = future["power_corrected"].values
        # Aanname: load is laag/constant
        P_load = 0.25

        T = len(P_solar)

        # Aantal mogelijke startmomenten
        num_possible_starts = T - duration_steps + 1
        if num_possible_starts <= 0:
            return SolarStatus.WAIT, None

        # --- CVXPY MODEL ---

        # 1. Binaire variabele: start op tijdstip t
        start_flags = cp.Variable(num_possible_starts, boolean=True)

        # 2. Variabele: Het daadwerkelijke DHW vermogen op elk tijdstip T
        # Dit is nu geen simpele boolean meer, maar een float variabele die we construeren
        dhw_power_vector = cp.Variable(T)

        constraints = []
        constraints.append(cp.sum(start_flags) == 1)  # Moet precies 1x starten

        # CONVOLUTIE / MATRIX OPBOUW
        # We bouwen de verwachte vermogenscurve op.
        # Als start_flags[k] == 1, dan wordt op [k...k+duur] het profiel 'geplakt'.

        profile_expr = 0
        for k in range(num_possible_starts):
            # Maak een vector van lengte T met overal nullen
            vec = np.zeros(T)
            # Plaats het power_profile op positie k
            vec[k : k + duration_steps] = power_profile

            # Tel op bij de expressie
            profile_expr += start_flags[k] * vec

        constraints.append(dhw_power_vector == profile_expr)

        # 3. Doelfunctie (Grid Minimalisatie)
        net_load = (P_load + dhw_power_vector) - P_solar
        grid_import = cp.pos(net_load)

        objective = cp.Minimize(cp.sum(grid_import))

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve()
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            return SolarStatus.WAIT, None

        # --- RESULTAAT ---
        best_start_idx = int(np.argmax(start_flags.value))
        planned_start = future.iloc[best_start_idx]["timestamp"]

        # Berekening voor context
        best_end_idx = best_start_idx + duration_steps
        slice_solar = P_solar[best_start_idx:best_end_idx]
        slice_dhw = power_profile  # Nu gebruiken we het echte profiel

        # Direct gebruik berekenen
        # Let op: slice_dhw en slice_solar moeten even lang zijn, dat is hier gegarandeerd
        direct_solar_usage = np.sum(np.minimum(slice_solar, slice_dhw + P_load))

        status = SolarStatus.WAIT
        minutes_to_start = (planned_start - current_time).total_seconds() / 60
        reason = f"Start gepland om {planned_start.strftime('%H:%M')}"

        if minutes_to_start <= 5:
            status = SolarStatus.START
            reason = "Starttijd bereikt. Optimalisatie voltooid."

        return status, SolarContext(
            actual_pv=0,  # In simulatie niet relevant
            energy_now=0,
            energy_best=round(direct_solar_usage * self.timestep_hours, 2),
            opportunity_cost=0,
            confidence=1.0,  # CVXPY is zeker van zijn zaak
            action=status,
            reason=reason,
            planned_start=planned_start,
            load_now=P_load,
        )
