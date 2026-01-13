import cvxpy as cp
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from context import SolarStatus, SolarContext

logger = logging.getLogger(__name__)

class Optimizer:
    def __init__(self, pv_max_kw: float, duration_hours: float, dhw_power_kw: float = 2.5):
        self.pv_max_kw = pv_max_kw
        self.duration_hours = duration_hours
        self.dhw_power_kw = dhw_power_kw # Hoeveel trekt de boiler? (bv 2.5kW)
        self.timestep_hours = 0.25       # Kwartierwaarden

    def optimize(self, df: pd.DataFrame, current_time: pd.Timestamp):
        """
        Vindt het beste startmoment met behulp van Mixed Integer Programming.
        """
        # Filter op toekomst
        future = df[df["timestamp"] >= current_time].copy().reset_index(drop=True)

        # We kijken max 24 uur vooruit (of de lengte van de dataframe)
        horizon_steps = min(len(future), int(24 / self.timestep_hours))
        future = future.iloc[:horizon_steps]

        if len(future) < (self.duration_hours / self.timestep_hours):
            logger.warning("Niet genoeg data voor optimalisatie.")
            return SolarStatus.WAIT, None

        # Data voorbereiden
        P_solar = future["power_corrected"].values  # Verwachte zon (kW)
        P_load  = 0.25 # Aanname baseload of future['projected_load'].values als je die hebt

        T = len(P_solar)
        duration_steps = int(self.duration_hours / self.timestep_hours)

        # --- CVXPY MODEL ---

        # Variabele: b[t] is een binaire variabele die '1' is als de DHW AAN staat op tijdstip t
        dhw_status = cp.Variable(T, boolean=True)

        # Variabele: start[t] is '1' op het moment dat de boiler START
        # Dit is nodig om te garanderen dat de boiler in één blok draait.
        # We hebben T - duration_steps + 1 mogelijke startmomenten.
        num_possible_starts = T - duration_steps + 1
        start_flags = cp.Variable(num_possible_starts, boolean=True)

        constraints = []

        # 1. Er moet precies 1 startmoment gekozen worden
        constraints.append(cp.sum(start_flags) == 1)

        # 2. Koppeling tussen startmoment en de dhw_status (convolutie matrix)
        # Als start_flags[k] == 1, dan moeten dhw_status[k ... k+duration] == 1 zijn
        # We bouwen dit op als een sommatie.
        status_expr = 0
        for k in range(num_possible_starts):
            # Maak een vector van lengte T die 1 is gedurende de duur vanaf k
            vec = np.zeros(T)
            vec[k : k + duration_steps] = 1
            status_expr += start_flags[k] * vec

        constraints.append(dhw_status == status_expr)

        # 3. Doelfunctie: Minimaliseer Grid Import
        # Grid Import = max(0, (Load + DHW - Solar))
        # Omdat max() convex is, kunnen we dit direct gebruiken.

        net_load = (P_load + (dhw_status * self.dhw_power_kw)) - P_solar
        grid_import = cp.pos(net_load) # cp.pos is max(0, x)

        # 1. Maak een 'Daylight Preference' curve
        # Dit is een simpele curve die 0 is in de nacht en oploopt naar 1 midden op de dag.
        # We baseren dit op de ruwe voorspelling, genormaliseerd.

        forecast_curve = future["power_corrected"].values
        max_val = np.max(forecast_curve)

        if max_val > 0.01:
            # Als er enig licht is, gebruik de vorm van de curve als 'voorkeur'
            preference_curve = forecast_curve / max_val
        else:
            # Als het model echt 0.0 zegt (midwinter nacht?), maak een default parabool over de horizon
            # Dit voorkomt dat de solver crasht of willekeurig kiest
            x_vals = np.linspace(-1, 1, T)
            preference_curve = np.maximum(0, 1 - x_vals**2) # Parabool 0 -> 1 -> 0

        # 2. De Doelfunctie
        # Primaire doel: Minimaliseer Grid Import (Groot gewicht: 1.0)
        # Secundaire doel: Maximaliseer draaien tijdens 'voorkeursmomenten' (Klein gewicht: 0.01)
        # Dit secundaire doel werkt alleen als 'tie-breaker' als Grid Import overal gelijk is.

        # We trekken de preference af (want we willen minimaliseren)
        # preference_score = sum(dhw_status * preference_curve)

        objective = cp.Minimize(
            cp.sum(grid_import) - (0.01 * cp.sum(cp.multiply(dhw_status, preference_curve)))
        )

        # Oplossen
        problem = cp.Problem(objective, constraints)

        # Kies solver. GLPK_MI of CBC is goed voor integers, maar standaard solvers werken vaak ook
        # voor simpele problemen of via brute-force heuristiek van CP.
        try:
            problem.solve()
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            return SolarStatus.WAIT, None

        # --- RESULTAAT VERWERKEN ---

        # Vind de start index
        start_vals = start_flags.value
        best_start_idx = int(np.argmax(start_vals))

        planned_start = future.iloc[best_start_idx]["timestamp"]
        minutes_to_start = (planned_start - current_time).total_seconds() / 60

        # Statistieken voor Context
        best_end_idx = best_start_idx + duration_steps

        # Bereken hoeveel zonne-energie we "vangen"
        slice_solar = P_solar[best_start_idx : best_end_idx]
        slice_load = np.full(len(slice_solar), P_load)
        slice_dhw = np.full(len(slice_solar), self.dhw_power_kw)

        # Self consumption tijdens de run
        # Min(Zon, DHW + Load) - Min(Zon, Load) = Wat DHW extra direct uit zon pakt
        direct_solar_usage = np.sum(np.minimum(slice_solar, slice_dhw + slice_load))
        total_energy_needed = self.dhw_power_kw * self.duration_hours

        solar_coverage_pct = (direct_solar_usage * self.timestep_hours) / total_energy_needed

        # Logica voor actie
        status = SolarStatus.WAIT
        reason = f"Start gepland om {planned_start.strftime('%H:%M')} ({int(solar_coverage_pct*100)}% zon dekking)"

        if minutes_to_start <= 5:
            status = SolarStatus.START
            reason = "Starttijd bereikt. Optimalisatie voltooid."

        return status, SolarContext(
            actual_pv=0, # In simulatie niet relevant
            energy_now=0,
            energy_best=round(direct_solar_usage * self.timestep_hours, 2),
            opportunity_cost=0,
            confidence=1.0, # CVXPY is zeker van zijn zaak
            action=status,
            reason=reason,
            planned_start=planned_start,
            load_now=P_load
        )