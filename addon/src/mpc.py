import cvxpy as cp
import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoilerConfig:
    volume_liters: float = 200.0
    power_kw: float = 2.2
    target_temp: float = 50.0
    min_temp: float = 30.0
    max_temp: float = 60.0
    loss_coef: float = 0.5
    deadline_hour: int = 17

    # Kosten
    grid_price: float = 0.22
    solar_price: float = 0.05  # Opportunity cost


class BoilerMPC:
    def __init__(self, config: BoilerConfig):
        self.cfg = config

    def solve(
        self, df_forecast: pd.DataFrame, current_temp: float, base_load_kw: float = 0.2
    ):
        df = df_forecast.copy().reset_index(drop=True)
        N = len(df)
        if N == 0:
            return None

        dt = (
            df["timestamp"].iloc[1] - df["timestamp"].iloc[0]
        ).total_seconds() / 3600.0

        # Physics
        kwh_per_degree = self.cfg.volume_liters * 0.001163
        temp_gain_per_kw = dt / kwh_per_degree
        temp_loss_step = self.cfg.loss_coef * dt

        # --- VARIABELEN ---
        T = cp.Variable(N + 1)
        P_boiler = cp.Variable(N)
        P_grid = cp.Variable(N, nonneg=True)

        # SLACK VARIABELEN (De "Redders in Nood")
        # Dit zijn variabelen die meten hoeveel we de regels overtreden
        slack_min = cp.Variable(N, nonneg=True)  # Hoeveel graden te koud (ondergrens)?
        slack_target = cp.Variable(nonneg=True)  # Hoeveel graden te koud (deadline)?

        # --- CONSTRAINTS ---
        constraints = []

        # 1. Startconditie
        constraints.append(T[0] == current_temp)

        excess_solar = (df["power_corrected"] - base_load_kw).clip(lower=0).values

        for t in range(N):
            # 2. Fysica
            constraints.append(
                T[t + 1] == T[t] + (P_boiler[t] * temp_gain_per_kw) - temp_loss_step
            )

            # 3. Boiler Grenzen
            constraints.append(P_boiler[t] >= 0)
            constraints.append(P_boiler[t] <= self.cfg.power_kw)

            # 4. Temperatuur Grenzen (SOFT CONSTRAINT)
            constraints.append(T[t + 1] + slack_min[t] >= self.cfg.min_temp)

            # Max temp mag hard blijven (veiligheid), of ook soft maken:
            constraints.append(T[t + 1] <= self.cfg.max_temp)

            # 5. Grid Balans
            constraints.append(P_grid[t] >= P_boiler[t] - excess_solar[t])

        # 6. Deadline (SOFT CONSTRAINT)
        deadline_rows = df[df["timestamp"].dt.hour == self.cfg.deadline_hour]
        if not deadline_rows.empty:
            idx_dl = deadline_rows.index[0]
            constraints.append(T[idx_dl] + slack_target >= self.cfg.target_temp)
        else:
            # Geen deadline gevonden (b.v. 's nachts na 17:00), dwing slack naar 0
            constraints.append(slack_target == 0)

        # --- OBJECTIVE ---

        # Kosten:
        SCALE = 10000.0

        # Kosten (x SCALE)
        cost_solar = cp.sum(P_boiler * self.cfg.solar_price) * SCALE
        cost_grid_premium = (
            cp.sum(P_grid * (self.cfg.grid_price - self.cfg.solar_price)) * SCALE
        )

        # Penalties (Laten we zo, die zijn al groot genoeg)
        penalty_min = cp.sum(slack_min) * 5000
        penalty_target = slack_target * 100000  # Enorme klap als deadline gemist wordt

        # Smoothness: Agressiever (x10) om 'geklapper' van 0.04kW tegen te gaan
        smoothness = 10.0 * cp.sum_squares(P_boiler[1:] - P_boiler[:-1])

        objective = cp.Minimize(
            cost_solar + cost_grid_premium + penalty_min + penalty_target + smoothness
        )

        # Oplossen
        prob = cp.Problem(objective, constraints)

        # Probeer OSQP, als die faalt ECOS, als die faalt SCS
        solvers = [cp.OSQP, cp.ECOS, cp.SCS]
        status = "failed"

        for solver in solvers:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    status = "success"
                    break
            except Exception:
                continue

        if status == "failed":
            logger.error("[MPC] All solvers failed. Fallback logic.")
            return None

        # Resultaten
        df["mpc_power_kw"] = P_boiler.value
        df["mpc_temp"] = T.value[:-1]

        # Debug info: Waarom stookt hij?
        # Als slack_min > 0, is het PANIEK (te koud).
        # Als slack_target > 0, halen we de deadline niet.

        df["boiler_status"] = (df["mpc_power_kw"] > 0.1).astype(int)

        return df
