import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# 1. SYSTEEM IDENTIFICATIE (RC, COP-vrij)
# =========================================================
class SystemIdentificator:
    """
    Identificeert R en C uitsluitend uit afkoelfases (WP uit).
    Geen COP nodig.
    """
    def __init__(self):
        self.R = 15.0   # K/kW
        self.C = 30.0   # kWh/K
        self.is_calibrated = False

    def calibrate(self, df: pd.DataFrame):
        if len(df) < 500:
            return

        df = df.sort_values("timestamp").copy()
        dt = 0.25

        # Alleen natuurlijke afkoeling
        cool = df[(df["wp_actual"] < 0.1) & (df["solar"] < 10)].copy()
        if len(cool) < 200:
            return

        cool["dT"] = cool["room_temp"].diff()
        cool["dT_io"] = (cool["room_temp"] - cool["outside"]) * dt

        X = cool[["dT_io"]].dropna()
        y = cool["dT"].loc[X.index]

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        # dT = -(dt/(R*C)) * (Tin-Tout)
        inv_RC = -model.coef_[0] / dt

        # schaal C conservatief, los R op
        self.C = np.clip(self.C, 15, 80)
        self.R = 1.0 / (inv_RC * self.C)

        self.is_calibrated = True
        logger.info(f"Identificatie: R={self.R:.2f} K/kW, C={self.C:.2f} kWh/K")

# =========================================================
# 2. COP-MODEL (GEEN HARDCODED CONSTANT)
# =========================================================
class COPModel:
    """
    Eenvoudig fysisch model:
    COP = a + b * T_out
    begrensd.
    """
    def __init__(self, a=2.5, b=0.08, cop_min=2.0, cop_max=5.0):
        self.a = a
        self.b = b
        self.cop_min = cop_min
        self.cop_max = cop_max

    def cop(self, T_out):
        return np.clip(self.a + self.b * T_out, self.cop_min, self.cop_max)

# =========================================================
# 3. ML RESIDUAL MODEL
# =========================================================
class MLResidualPredictor:
    """
    Leert rest-dT per stap (K/15min) bovenop RC-model.
    """
    def __init__(self, path: str, target_name: str):
        self.path = Path(path)
        self.target_name = target_name
        self.model = None
        if self.path.exists():
            self.model = joblib.load(self.path)

    def train(self, df: pd.DataFrame, R, C, cop_model: COPModel, is_dhw=False):
        dt = 0.25
        # RC voorspelling (zonder ML)
        cop = cop_model.cop(df["outside"])

        if not is_dhw:
            # Kamer: winst - verlies
            dT_rc = ((df["wp_actual"] * cop) - (df["room_temp"] - df["outside"]) / R) * dt / C
            target_val = df["room_temp"]
        else:
            # Boiler: alleen winst (verlies is minimaal/verwerkt in residuals)
            dT_rc = (df["wp_actual"] * cop * dt) / 10.0
            target_val = df["dhw_temp"]

        y = target_val.diff() - dT_rc
        X = df[["solar", "wind", "outside"]].fillna(0)

        self.model = HistGradientBoostingRegressor()
        self.model.fit(X.iloc[1:], y.iloc[1:])
        joblib.dump(self.model, self.path)
        logger.info(f"ML Model {self.target_name} getraind.")

    def predict(self, forecast_df):
        if self.model is None:
            return np.zeros(len(forecast_df))
        return self.model.predict(
            forecast_df[["solar", "wind", "temp"]].fillna(0)
        )

# =========================================================
# 4. MPC
# =========================================================
class ThermalMPC:
    def __init__(self, ident: SystemIdentificator, cop_vwv: COPModel, cop_dhw: COPModel):
        self.ident = ident
        self.cop_vwv = cop_vwv
        self.cop_dhw = cop_dhw
        self.horizon = 48
        self.p_el_max = 3.5

    def solve(self, state, forecast_df, prices, vwv_residuals, dhw_residuals):
        """
        state: {'room_temp', 'dhw_top', 'dhw_bottom'}
        forecast_df: DF met 'temp', 'pv_forecast', 'house_load'
        prices: array van energieprijzen
        vwv_residuals: lijst met ML correcties voor de kamer
        dhw_residuals: lijst met ML correcties voor de boiler (bijv. verbruik)
        """
        T = self.horizon
        dt = 0.25 # Kwartier

        # Fysische Parameters uit de identificatie
        R, C = self.ident.R, self.ident.C

        # --- BOILER CONSTANTEN (200 Liter) ---
        # Energie om 200L water 1 graad te verwarmen is ~0.232 kWh
        # We gebruiken dit als deler voor de temperatuurstijging.
        boiler_mass_factor = 0.232
        dhw_standby_loss = 0.04 # Graden verlies per kwartier (isolatievat)

        # 1. VARIABELEN (Lineair & Binair)
        u_vwv = cp.Variable(T, boolean=True)  # Modus VWV
        u_dhw = cp.Variable(T, boolean=True)  # Modus DHW
        p_el_vwv = cp.Variable(T, nonneg=True) # Elektrisch vermogen voor vloer
        p_el_dhw = cp.Variable(T, nonneg=True) # Elektrisch vermogen voor boiler

        t_room = cp.Variable(T + 1)
        t_dhw = cp.Variable(T + 1)

        # Slack variabelen (Voorkomen 'Infeasible' fouten bij grensoverschrijding)
        slack_room_low = cp.Variable(T, nonneg=True)
        slack_dhw_low = cp.Variable(T, nonneg=True)

        # 2. CONSTRAINTS
        constraints = [
            t_room[0] == state["room_temp"],
            t_dhw[0] == (state["dhw_top"] + state["dhw_bottom"]) / 2
        ]

        target_dhw=50.0

        for t in range(T):
            # Exclusiviteit: WP kan niet tegelijkertijd VWV en DHW doen (driewegklep)
            constraints += [u_vwv[t] + u_dhw[t] <= 1]

            # Vermogensbegrenzing gekoppeld aan binaire status
            constraints += [p_el_vwv[t] <= self.p_el_max * u_vwv[t]]
            constraints += [p_el_dhw[t] <= self.p_el_max * u_dhw[t]]

            # Minimale vermogens (50% van max bij aan)
            constraints += [p_el_vwv[t] >= 0.5 * u_vwv[t]]
            constraints += [p_el_dhw[t] >= 0.5 * u_dhw[t]]

            # Haal COP op voor dit tijdstip (temperatuurafhankelijk)
            c_vwv = self.cop_vwv.cop(forecast_df["temp"].iloc[t])
            c_dhw = self.cop_dhw.cop(forecast_df["temp"].iloc[t])

            # --- KAMER DYNAMICA ---
            p_th_room = p_el_vwv[t] * c_vwv
            loss = (t_room[t] - forecast_df["temp"].iloc[t]) / R
            # Nieuwe temp = huidige + (winst - verlies) + ML_correctie
            constraints += [
                t_room[t+1] == t_room[t] + (p_th_room - loss) * dt / C + vwv_residuals[t]
            ]

            # --- BOILER DYNAMICA ---
            p_th_dhw = p_el_dhw[t] * c_dhw
            # Nieuwe temp = huidige + (winst / massa) - stilstandsverlies + ML_correctie (verbruik)
            constraints += [
                t_dhw[t+1] == t_dhw[t] + (p_th_dhw * dt) / boiler_mass_factor - dhw_standby_loss + dhw_residuals[t]
            ]

            # --- COMFORT GRENZEN (Soft) ---
            # t_room + slack >= 19.5 (Als t_room 19.0 is, wordt slack 0.5 en betaal je een boete)
            constraints += [t_room[t] + slack_room_low[t] >= 19.5]
            constraints += [t_dhw[t] + slack_dhw_low[t] >= 35.0]

            # Harde grenzen (Veiligheid)
            constraints += [t_room[t] <= 24.0, t_dhw[t] <= 60.0]

            # Supply temperature proxy: voorkom dat de vloer te heet wordt (max vermogen bij lage kamer-T)
            constraints += [t_room[t] + p_el_vwv[t] * 3.0 <= 40]

        # 3. OBJECTIVE (Kosten + Slijtage + Comfort + Boetes)
        # Netto import (Import - Export wordt meegerekend via de positieve kant van de balans)
        net_load = (p_el_vwv + p_el_dhw + forecast_df["house_load"].values - forecast_df["pv_forecast"].values)
        cost = cp.sum(cp.multiply(cp.pos(net_load), prices))

        # Anti-pendel: Straf het omschakelen of aan/uit gaan (MILP switches)
        switches = cp.sum(cp.abs(u_vwv[1:] - u_vwv[:-1])) + cp.sum(cp.abs(u_dhw[1:] - u_dhw[:-1]))

        # Comfort: probeer de kamer op 20.5 graden te houden
        comfort_tracking = cp.sum(cp.abs(t_room - 20.5)) * 0.1

        # Slack boete: Zeer hoog om comfortgrenzen te bewaken
        violation_penalty = cp.sum(slack_room_low + slack_dhw_low) * 150.0

        dhw_comfort = cp.abs(t_dhw[T] - target_dhw) * 5.0

        # Urgentie bonus voor DHW (als top sensor koud is, verdien je een "korting" door te verwarmen)
        # Dit is de zachte vervanger van de harde u_dhw[0] == 1 constraint
        dhw_urgent_bonus = 0
        if state["dhw_top"] < 30:
            dhw_urgent_bonus = u_dhw[0] * -100.0

        # Totaal te minimaliseren
        objective = cp.Minimize(cost + 0.5 * switches + comfort_tracking + violation_penalty + dhw_comfort + dhw_urgent_bonus)

        # 4. SOLVE
        problem = cp.Problem(objective, constraints)

        try:
            # CBC via CyLP is de aanbevolen MILP solver
            problem.solve(solver=cp.CBC, verbose=False)
        except:
            logger.warning("CBC solver niet beschikbaar, probeer andere MILP solvers.")
            # Fallback naar andere beschikbare MILP solvers (GLPK, SCIP)
            problem.solve()

        # Foutafhandeling
        if u_vwv.value is None:
            logger.error(f"Solver Status: {problem.status}")
            raise RuntimeError(f"MILP solver kon geen oplossing vinden. Status: {problem.status}")

        # 5. RESULTATEN VERWERKEN
        mode = "DHW" if u_dhw.value[0] > 0.5 else "VWV" if u_vwv.value[0] > 0.5 else "OFF"
        current_p_el = float(p_el_vwv.value[0] + p_el_dhw.value[0])

        # State of Charge (SoC) berekening voor logging
        soc = (t_dhw.value[0] - 35) / (55 - 35)

        return {
            "mode": mode,
            "target_power": round(current_p_el, 3),
            "planned_room": t_room.value.tolist(),
            "planned_dhw": t_dhw.value.tolist(),
            "soc_dhw": max(0, min(1, soc)),
            "status": problem.status
        }

# =========================================================
# 5. EMS
# =========================================================
class EnergyManagementSystem:
    def __init__(self):
        self.ident = SystemIdentificator()

        # Voor Vloerverwarming (Lage temperatuur = Hoge efficiëntie)
        self.cop_vwv = COPModel(a=3.0, b=0.08, cop_min=2.0, cop_max=5)

        # Voor Warm Water (Hoge temperatuur = Lage efficiëntie)
        self.cop_dhw = COPModel(a=2.0, b=0.06, cop_min=1.5, cop_max=3.0)

        # APARTE ML MODELLEN
        self.vwv_res = MLResidualPredictor("res_vwv.pkl", "Kamer")
        self.dhw_res = MLResidualPredictor("res_dhw.pkl", "Boiler")

        self.mpc = ThermalMPC(self.ident, self.cop_vwv, self.cop_dhw)

    def step(self, state, history_df, forecast_df, prices):
        if not self.ident.is_calibrated:
            self.ident.calibrate(history_df)

        # Voorspel voor beide systemen de residuals
        vwv_residuals = self.vwv_res.predict(forecast_df)
        dhw_residuals = self.dhw_res.predict(forecast_df)

        return self.mpc.solve(state, forecast_df, prices, vwv_residuals, dhw_residuals)

# =========================================================
# VOORBEELD VAN GEBRUIK (MOCK DATA)
# =========================================================
if __name__ == "__main__":
    # Mock data voor 48 kwartieren (12 uur)
    forecast = pd.DataFrame({
        'temp': np.linspace(5, 10, 48),
        'pv_forecast': np.maximum(0, np.sin(np.linspace(0, np.pi, 48)) * 4.0),
        'wind': np.random.uniform(2, 8, 48),
        'house_load': np.full(48, 0.4)
    })
    prices = np.random.uniform(0.10, 0.35, 48) # Dynamische prijzen

    ems = EnergyManagementSystem()

    metingen = {
        'room_temp': 23.0,
        'dhw_top': 45.0,
        'dhw_bottom': 43.0
    }

    # In een echte situatie zou history_df uit je database komen
    history_mock = pd.DataFrame(columns=['timestamp', 'room_temp', 'outside', 'wp_actual', 'solar'])

    besluit = ems.step(metingen, history_mock, forecast, prices)
    print(f"Besluit {besluit['status']}:")
    print(f"Besluit van de MPC: {besluit['mode']} op {besluit['target_power']:.2f} kW")
    print(f"Huidige Boiler SoC: {besluit['soc_dhw']*100:.1f}%")

    print(f"Gepland kamerverloop (komende 2 uur):")
    # t_room heeft T+1 waarden, dus index 0 t/m 8 zijn de eerste 2 uur (8 kwartieren)
    for i, temp in enumerate(besluit['planned_room'][:9]):
        print(f"  T + {i*15}m: {temp:.2f} °C")

    print(f"Gepland boiler (komende 2 uur):")
    # t_room heeft T+1 waarden, dus index 0 t/m 8 zijn de eerste 2 uur (8 kwartieren)
    for i, temp in enumerate(besluit['planned_dhw'][:9]):
        print(f"  T + {i*15}m: {temp:.2f} °C")