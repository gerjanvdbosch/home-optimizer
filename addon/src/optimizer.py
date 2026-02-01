import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging
import os

from datetime import datetime, timedelta
from config import Config
from context import Context
from database import Database
from utils import add_cyclic_time_features
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

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

    def __init__(self, path):
        self.R = 15.0  # K/kW
        self.C = 30.0  # kWh/K
        self.feature_cols = ["room_temp", "temp", "wp_actual", "pv_actual"]
        self.path = Path(path)
        self.is_fitted = False
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.R = data["R"]
                self.C = data["C"]
                self.is_fitted = True
                logger.info(
                    f"[Optimizer] Identificatie: R={self.R:.2f} K/kW, C={self.C:.2f} kWh/K"
                )
            except Exception:
                logger.error("[Optimizer] Model corrupt.")

    def train(self, df: pd.DataFrame):
        df = df.copy()
        df = df.set_index("timestamp")
        df = df.sort_index()
        df = df.resample("15min").asfreq()
        dt = 0.25  # Kwartier

        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # WP uit, Zon uit, Binnen warmer dan Buiten
        cool = df[
            (df["wp_actual"] < 0.1)
            & (df["pv_actual"] < 0.05)
            & (df["room_temp"] > df["temp"])
        ].copy()

        # 2. FEATURE ENGINEERING
        cool["dT"] = cool["room_temp"].diff()

        # We stoppen dt al in X
        cool["dT_io"] = (cool["room_temp"] - cool["temp"]) * dt

        # Verwijder NaN en positieve dT (opwarming door interne winst negeren)
        train_data = cool[["dT_io", "dT"]].dropna()
        train_data = train_data[train_data["dT"] < 0]

        if len(train_data) < 25:
            logger.info(
                f"[Optimizer] Niet genoeg data voor identificatie ({len(train_data)} samples)."
            )
            return

        X = train_data[["dT_io"]]
        y = train_data["dT"]

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        coef = model.coef_[0]

        # 3. BEREKENING R (Correcte formule)
        # Formule: coef = -1 / (R * C)
        # Dus: R = -1 / (coef * C)

        # Check of coefficient logisch is (moet negatief zijn voor afkoeling)
        if coef >= 0:
            return

        calculated_R = -1.0 / (coef * self.C)

        # 4. CHECKS & OPSLAAN
        if 2.0 < calculated_R < 100.0:
            self.R = calculated_R
            self.is_fitted = True
            joblib.dump({"R": self.R, "C": self.C}, self.path)
            logger.info(f"[Optimizer] R={self.R:.2f} K/kW, C={self.C:.2f} kWh/K")


# =========================================================
# 2. COP-MODEL
# =========================================================
class COPModel:
    def __init__(self, t_supply_target, efficiency=0.50, cop_min=1.0, cop_max=5.0):
        self.t_supply = t_supply_target
        self.eta = efficiency
        self.cop_min = cop_min
        self.cop_max = cop_max

    def cop(self, T_out):
        # Voorkom divide by zero als T_out == T_supply (onwaarschijnlijk maar toch)
        delta_t = np.maximum(self.t_supply - T_out, 5.0)

        # T in Kelvin
        t_h_kelvin = self.t_supply + 273.15

        cop_theoretical = t_h_kelvin / delta_t
        cop_real = cop_theoretical * self.eta

        return np.clip(cop_real, self.cop_min, self.cop_max)


# =========================================================
# 3. ML RESIDUAL MODEL
# =========================================================
class MLResidualPredictor:
    def __init__(self, path: str):
        self.path = Path(path)
        self.model = None
        # Definieer vaste feature-sets om mismatches te voorkomen
        self.features_ufh = [
            "temp",
            "solar",
            "wind",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
        ]
        self.features_dhw = ["temp", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

        if self.path.exists():
            self.model = joblib.load(self.path)

    def train(self, df: pd.DataFrame, R, C, cop_model, is_dhw=False):
        df = df.copy()
        df = df.set_index("timestamp")
        df = df.sort_index()
        df = df.resample("15min").asfreq()
        dt = 0.25
        boiler_mass_factor = 0.232

        # Voeg 'hour' toe aan de bron-dataframe
        df = add_cyclic_time_features(df, col_name="timestamp")

        # 1. Bereken de theoretische RC-delta (Physics Baseline)
        outside = df["temp"]
        wp = df["wp_actual"]
        cop = cop_model.cop(outside)

        if not is_dhw:
            df["solar"] = df["pv_actual"]
            temp = df["room_temp"]
            dT_rc = ((wp * cop) - (temp - outside) / R) * dt / C
            y_actual = df["room_temp"].shift(-1) - df["room_temp"]
            feature_cols = self.features_ufh
        else:
            dT_rc = (wp * cop * dt) / boiler_mass_factor
            y_actual = df["dhw_temp"].shift(-1) - df["dhw_temp"]
            feature_cols = self.features_dhw

        target_series = (y_actual - dT_rc).rename("target_residual")

        X = df.reindex(columns=feature_cols)
        train_df = pd.concat([X, target_series], axis=1).dropna()

        if len(train_df) > 50:
            X_train = train_df[feature_cols]
            y_train = train_df["target_residual"]

            self.model = HistGradientBoostingRegressor()
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.path)
            logger.info(
                f"[Optimizer] Model {self.path} getraind op {len(X_train)} samples."
            )

    def predict(self, forecast_df, is_dhw=False):
        """
        Voorspelt residuals voor de horizon.
        LET OP: forecast_df moet de juiste kolomnamen bevatten.
        """
        if self.model is None:
            return np.zeros(len(forecast_df))

        # 1. Voorbereiden van de features
        df_input = forecast_df.copy()
        df_input["solar"] = df_input["power_corrected"]
        df_input = add_cyclic_time_features(df_input, col_name="timestamp")

        # 2. Selecteer de juiste features voor het model
        feature_cols = self.features_dhw if is_dhw else self.features_ufh

        # 3. Voorspel
        X_predict = df_input.reindex(columns=feature_cols, fill_value=0.0)

        return self.model.predict(X_predict)


# =========================================================
# 4. MPC
# =========================================================
class ThermalMPC:
    def __init__(
        self, ident: SystemIdentificator, cop_ufh: COPModel, cop_dhw: COPModel
    ):
        self.ident = ident
        self.cop_ufh = cop_ufh
        self.cop_dhw = cop_dhw
        self.horizon = 48
        self.p_el_max = 3.5
        self.dhw_target = 50.0
        self.dhw_min = 35.0
        self.cold_water = 15.0

    def solve(self, state, forecast_df, prices, ufh_residuals, dhw_residuals):
        """
        state: {'room_temp', 'dhw_top', 'dhw_bottom'}
        forecast_df: DF met 'temp', 'power_corrected', 'load_corrected'
        prices: array van energieprijzen
        ufh_residuals: lijst met ML correcties voor de kamer
        dhw_residuals: lijst met ML correcties voor de boiler (bijv. verbruik)
        """
        T = self.horizon
        dt = 0.25  # Kwartier

        # Fysische Parameters uit de identificatie
        R, C = self.ident.R, self.ident.C

        # --- BOILER CONSTANTEN (200 Liter) ---
        # Energie om 200L water 1 graad te verwarmen is ~0.232 kWh
        # We gebruiken dit als deler voor de temperatuurstijging.
        boiler_mass_factor = 0.232
        dhw_standby_loss = 0.04  # Graden verlies per kwartier (isolatievat)

        # 1. VARIABELEN (Lineair & Binair)
        u_ufh = cp.Variable(T, boolean=True)  # Modus UFH
        u_dhw = cp.Variable(T, boolean=True)  # Modus DHW
        p_el_ufh = cp.Variable(T, nonneg=True)  # Elektrisch vermogen voor vloer
        p_el_dhw = cp.Variable(T, nonneg=True)  # Elektrisch vermogen voor boiler

        t_room = cp.Variable(T + 1)
        t_dhw = cp.Variable(T + 1)

        # Slack variabelen (Voorkomen 'Infeasible' fouten bij grensoverschrijding)
        slack_room_low = cp.Variable(T, nonneg=True)
        slack_dhw_low = cp.Variable(T, nonneg=True)

        # 2. CONSTRAINTS
        constraints = [
            t_room[0] == state["room_temp"],
            t_dhw[0] == (state["dhw_top"] * 0.7 + state["dhw_bottom"] * 0.3),
        ]

        for t in range(T):
            # Exclusiviteit: WP kan niet tegelijkertijd ufh en DHW doen (driewegklep)
            constraints += [u_ufh[t] + u_dhw[t] <= 1]

            # Vermogensbegrenzing gekoppeld aan binaire status
            constraints += [p_el_ufh[t] <= self.p_el_max * u_ufh[t]]
            constraints += [p_el_dhw[t] <= self.p_el_max * u_dhw[t]]

            # Minimale vermogens (50% van max bij aan)
            constraints += [p_el_ufh[t] >= 0.5 * u_ufh[t]]
            constraints += [p_el_dhw[t] >= 0.5 * u_dhw[t]]

            # Haal COP op voor dit tijdstip (temperatuurafhankelijk)
            c_ufh = self.cop_ufh.cop(forecast_df["temp"].iloc[t])
            c_dhw = self.cop_dhw.cop(forecast_df["temp"].iloc[t])

            # --- KAMER DYNAMICA ---
            p_th_room = p_el_ufh[t] * c_ufh
            loss = (t_room[t] - forecast_df["temp"].iloc[t]) / R
            # Nieuwe temp = huidige + (winst - verlies) + ML_correctie
            constraints += [
                t_room[t + 1]
                == t_room[t] + (p_th_room - loss) * dt / C + ufh_residuals[t]
            ]

            # --- BOILER DYNAMICA ---
            p_th_dhw = p_el_dhw[t] * c_dhw
            # Nieuwe temp = huidige + (winst / massa) - stilstandsverlies + ML_correctie (verbruik)
            constraints += [
                t_dhw[t + 1]
                == t_dhw[t]
                + (p_th_dhw * dt) / boiler_mass_factor
                - dhw_standby_loss
                + dhw_residuals[t]
            ]

            # Max 4 graden per kwartier (voorkomt raket-boiler)
            constraints += [t_dhw[t + 1] - t_dhw[t] <= 4.0]

            # --- COMFORT GRENZEN (Soft) ---
            # t_room + slack >= 19.5 (Als t_room 19.0 is, wordt slack 0.5 en betaal je een boete)
            constraints += [t_room[t] + slack_room_low[t] >= 19.5]
            constraints += [t_dhw[t] + slack_dhw_low[t] >= self.dhw_min]

            # Harde grenzen (Veiligheid)
            constraints += [t_room[t] <= 24.0, t_dhw[t] <= 60.0]

            # Supply temperature proxy: voorkom dat de vloer te heet wordt (max vermogen bij lage kamer-T)
            constraints += [t_room[t] + p_el_ufh[t] * 4.0 <= 40]

        # 3. OBJECTIVE (Kosten + Slijtage + Comfort + Boetes)
        # Netto import (Import - Export wordt meegerekend via de positieve kant van de balans)
        net_load = (
            p_el_ufh
            + p_el_dhw
            + forecast_df["load_corrected"].values
            - forecast_df["power_corrected"].values
        )
        cost = cp.sum(cp.multiply(cp.pos(net_load), prices)) * dt

        # Anti-pendel: Straf het omschakelen of aan/uit gaan (MILP switches)
        switches = cp.sum(cp.abs(u_ufh[1:] - u_ufh[:-1])) + cp.sum(
            cp.abs(u_dhw[1:] - u_dhw[:-1])
        )

        # Comfort: probeer de kamer op 20.5 graden te houden
        comfort_tracking = cp.sum(cp.abs(t_room - 20.5)) * 0.1

        # Slack boete: Zeer hoog om comfortgrenzen te bewaken
        violation_penalty = cp.sum(slack_room_low + slack_dhw_low) * 150.0

        dhw_comfort = cp.abs(t_dhw[T] - self.dhw_target) * 5.0

        # Totaal te minimaliseren
        objective = cp.Minimize(
            cost + 0.5 * switches + comfort_tracking + violation_penalty + dhw_comfort
        )

        # 4. SOLVE
        problem = cp.Problem(objective, constraints)

        try:
            # CBC via CyLP is de aanbevolen MILP solver
            problem.solve(
                solver=cp.CBC, verbose=LOG_LEVEL == "DEBUG", maximumSeconds=15
            )
        except Exception as e:
            logger.warning(
                f"[Optimizer] CBC solver niet beschikbaar, probeer andere MILP solvers. Fout: {e}"
            )
            # Fallback naar andere beschikbare MILP solvers (GLPK, SCIP)
            problem.solve(verbose=LOG_LEVEL == "DEBUG", maximumSeconds=15)

        # Foutafhandeling
        if u_ufh.value is None:
            logger.error(
                f"[Optimizer] MILP solver kon geen oplossing vinden (status={problem.status})"
            )

            return

        # 5. RESULTATEN VERWERKEN
        mode = (
            "DHW" if u_dhw.value[0] > 0.5 else "UFH" if u_ufh.value[0] > 0.5 else "OFF"
        )
        current_p_el = float(p_el_ufh.value[0] + p_el_dhw.value[0])

        dhw_avg = (state["dhw_top"] + state["dhw_bottom"]) / 2

        # Bereken SoC
        soc = (dhw_avg - self.cold_water) / (self.dhw_target - self.cold_water)

        # kWh berekenen voor de optimizer
        dhw_energy_kwh = max(0.0, (dhw_avg - self.cold_water) * boiler_mass_factor)

        return {
            "mode": mode,
            "target_power": round(current_p_el, 3),
            "planned_room": t_room.value.tolist(),
            "planned_dhw": t_dhw.value.tolist(),
            "dhw_soc": max(0, min(1, soc)),
            "dhw_energy_kwh": dhw_energy_kwh,
            "cost_projected": float(cost.value),
            "status": problem.status,
        }


# =========================================================
# 5. EMS
# =========================================================
class Optimizer:
    def __init__(self, config: Config, database: Database):
        self.database = database
        self.ident = SystemIdentificator(config.rc_model_path)

        self.cop_ufh = COPModel(
            t_supply_target=27.0, efficiency=0.50, cop_min=2.5, cop_max=5.0
        )
        self.cop_dhw = COPModel(
            t_supply_target=55.0, efficiency=0.45, cop_min=1.5, cop_max=3.0
        )

        # APARTE ML MODELLEN
        self.ufh_res = MLResidualPredictor(config.ufh_model_path)
        self.dhw_res = MLResidualPredictor(config.dhw_model_path)

        self.mpc = ThermalMPC(self.ident, self.cop_ufh, self.cop_dhw)

    def resolve(self, context: Context):
        horizon_df = context.forecast_df.iloc[: self.mpc.horizon].copy()

        # Voorspel voor beide systemen de residuals
        ufh_residuals = self.ufh_res.predict(horizon_df, is_dhw=False)
        dhw_residuals = self.dhw_res.predict(horizon_df, is_dhw=True)

        state = {
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        # Vaste prijs voor testdoeleinden
        prices = [0.21] * len(horizon_df)

        return self.mpc.solve(state, horizon_df, prices, ufh_residuals, dhw_residuals)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)

        # Haal samengevoegde data (Measurements + Forecast)
        history_df = self.database.get_history(cutoff_date=cutoff)

        self.ident.train(history_df)

        # Train ML modellen
        self.ufh_res.train(
            history_df, self.ident.R, self.ident.C, self.cop_ufh, is_dhw=False
        )
        self.dhw_res.train(
            history_df, self.ident.R, self.ident.C, self.cop_dhw, is_dhw=True
        )
