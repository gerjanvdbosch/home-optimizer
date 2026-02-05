import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging

from datetime import datetime, timedelta
from config import Config
from context import Context
from database import Database
from utils import add_cyclic_time_features
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# 1. COP-MODEL (Carnot-based)
# =========================================================
class COPModel:
    def __init__(self, efficiency=0.45, cop_min=1.0, cop_max=6.0):
        self.eta = efficiency
        self.cop_min = cop_min
        self.cop_max = cop_max

    def get_cop(self, T_out, T_sink):
        """
        Berekent COP o.b.v. buitentemperatuur en doeltemperatuur (sink).
        T_sink is de watertemperatuur (bijv. 30C voor Vloer of 50C voor Boiler).
        """
        # Voeg 3 graden toe aan T_sink voor de warmtewisselaar delta (condensor)
        T_sink_k = T_sink + 273.15 + 3.0
        T_out_k = T_out + 273.15

        # Carnot limiet: T_sink / (T_sink - T_out)
        # We gebruiken een minimum delta van 5 graden om deling door nul te voorkomen
        delta_t = np.maximum(T_sink_k - T_out_k, 5.0)

        cop_theoretical = T_sink_k / delta_t
        cop_real = cop_theoretical * self.eta

        return np.clip(cop_real, self.cop_min, self.cop_max)


# =========================================================
# 2. SYSTEEM IDENTIFICATIE (RC Model)
# =========================================================
class SystemIdentificator:
    """
    Identificeert R (Isolatiewaarde) en C (Warmteopslagcapaciteit) van de woning.
    """

    def __init__(self, path):
        self.R = 15.0  # Default: K/kW
        self.C = 30.0  # Default: kWh/K
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
                    f"[Identificatie] Geladen: R={self.R:.2f} K/kW, C={self.C:.2f} kWh/K"
                )
            except Exception:
                logger.error("[Identificatie] Model bestand corrupt of verouderd.")

    def train(self, df: pd.DataFrame):
        """
        Traint het RC model op historische data.
        Filtert zonlicht en outliers weg voor een zuivere fysieke identificatie.
        """
        df = df.copy()
        df = df.set_index("timestamp").sort_index()
        # Vul kleine gaatjes op
        df = df.resample("15min").interpolate(method="linear", limit=2)

        # 1. Bepaal de temperatuurverandering over de KOMENDE 60 minuten
        df["dT_1h"] = df["room_temp"].shift(-4) - df["room_temp"]

        # 2. Schat de COP voor het historische thermische vermogen
        # Aanname: Vloerverwarming water is kamer_temp + 6 graden
        cop_calc = COPModel(efficiency=0.50)
        df["cop_actual"] = df.apply(
            lambda row: cop_calc.get_cop(row["temp"], row["room_temp"] + 6.0), axis=1
        )

        # Als wp_actual elektrisch is (kW), dan vermenigvuldigen met COP
        df["p_th_actual"] = df["wp_actual"] * df["cop_actual"]

        # 3. Features voorbereiden (Rolling mean om ruis te onderdrukken)
        # X1: Verlies naar buiten (T_room - T_out)
        loss_potential = -(df["room_temp"] - df["temp"])
        df["X1"] = loss_potential.rolling(window=4).mean().shift(-3)

        # X2: Thermische winst
        df["X2"] = df["p_th_actual"].rolling(window=4).mean().shift(-3)

        # 4. Filteren
        # Geen zon (pv < 0.05) en stabiele metingen (dT < 1.5)
        mask = (df["pv_actual"] < 0.05) & (df["dT_1h"].abs() < 1.5)
        train_data = df[mask][["X1", "X2", "dT_1h"]].dropna()

        if len(train_data) < 50:
            logger.warning(
                "[Identificatie] Te weinig stabiele data voor betrouwbare training."
            )
            return

        X = train_data[["X1", "X2"]]
        y = train_data["dT_1h"]

        # 5. Regressie
        # dT = (P_th / C) - (deltaT / RC)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        coef_loss, coef_gain = model.coef_

        # 6. Begrenzing (Clamping) om fysiek onmogelijke waarden te voorkomen
        # C tussen 15 en 120 kWh/K
        c_gain_clamped = np.clip(coef_gain, 1.0 / 120.0, 1.0 / 15.0)
        self.C = 1.0 / c_gain_clamped

        # R tussen 5 en 40 K/kW
        r_loss_clamped = np.clip(coef_loss, 1.0 / (40.0 * self.C), 1.0 / (5.0 * self.C))
        self.R = 1.0 / (r_loss_clamped * self.C)

        self.is_fitted = True
        joblib.dump({"R": self.R, "C": self.C}, self.path)
        logger.info(f"[Identificatie] Nieuw getraind: R={self.R:.2f}, C={self.C:.2f}")


# =========================================================
# 3. ML RESIDUAL MODEL
# =========================================================
class MLResidualPredictor:
    """
    Voorspelt het verschil (residual) tussen het RC-model en de werkelijkheid.
    Vangt effecten op van Zon, Wind, Douchen, Koken, etc.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.model = None
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
            try:
                self.model = joblib.load(self.path)
            except Exception:
                logger.error(f"Kon model {path} niet laden.")

    def train(self, df: pd.DataFrame, R, C, cop_model, is_dhw=False):
        df = df.copy()
        df = df.set_index("timestamp").sort_index()
        df = df.resample("15min").interpolate(method="linear", limit=2)
        df = df.reset_index()
        df = add_cyclic_time_features(df, col_name="timestamp")

        dt = 0.25
        boiler_mass_factor = 0.232  # 200L boiler

        if not is_dhw:
            # UFH: Sink ~26C (bij 20C kamer)
            df["cop_calc"] = df.apply(
                lambda row: cop_model.get_cop(row["temp"], row["room_temp"] + 6.0),
                axis=1,
            )

            # Theoretisch: dT = (Winst - Verlies) / Massa
            loss = (df["room_temp"] - df["temp"]) / R
            dT_rc = ((df["wp_actual"] * df["cop_calc"]) - loss) * dt / C
            y_actual = df["room_temp"].shift(-1) - df["room_temp"]

            df["solar"] = df["pv_actual"]
            feature_cols = self.features_ufh
        else:
            # DHW
            df["dhw_temp"] = (df["dhw_top"] + df["dhw_bottom"]) / 2

            # Sink ~45C-55C. We nemen hier gemiddeld +9 tov water
            df["cop_calc"] = df.apply(
                lambda row: cop_model.get_cop(row["temp"], row["dhw_temp"] + 9.0),
                axis=1,
            )

            dT_rc = (df["wp_actual"] * df["cop_calc"] * dt) / boiler_mass_factor
            y_actual = df["dhw_temp"].shift(-1) - df["dhw_temp"]
            feature_cols = self.features_dhw

        target = (y_actual - dT_rc).rename("target_residual")

        train_df = pd.concat([df[feature_cols], target], axis=1).dropna()

        if len(train_df) > 50:
            self.model = HistGradientBoostingRegressor()
            self.model.fit(train_df[feature_cols], train_df["target_residual"])
            joblib.dump(self.model, self.path)
            logger.info(f"[ML] Model {self.path.name} getraind.")

    def predict(self, forecast_df, is_dhw=False):
        if self.model is None:
            return np.zeros(len(forecast_df))

        df_in = forecast_df.copy()
        df_in["solar"] = df_in["power_corrected"]  # Gebruik forecast solar
        df_in = add_cyclic_time_features(df_in, col_name="timestamp")

        cols = self.features_dhw if is_dhw else self.features_ufh
        X = df_in.reindex(columns=cols, fill_value=0.0)

        return self.model.predict(X)


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

            # --- DHW VERMOGEN: Geen 50%, maar VOL vermogen (of bijv. 80-100%) ---
            # Voor DHW willen we vaak dat de WP op een vast, efficiënt hoog vermogen draait.
            constraints += [p_el_dhw[t] == self.p_el_max * u_dhw[t]]

            # --- UFH VERMOGEN: Minimaal 0.8 kW en afhankelijk van buitentemperatuur ---
            # We berekenen een minimale ondergrens: 0.8 kW basis,
            # plus extra vermogen als het buiten erg koud is (bijv. +0.1kW per 5 graden onder nul)
            temp_correction = cp.maximum(0, (0 - forecast_df["temp"].iloc[t]) * 0.05)
            min_p_ufh = (0.8 + temp_correction) * u_ufh[t]

            constraints += [p_el_ufh[t] >= min_p_ufh]
            constraints += [p_el_ufh[t] <= self.p_el_max * u_ufh[t]]

            # Haal COP op voor dit tijdstip (temperatuurafhankelijk)
            c_ufh = self.cop_ufh.get_cop(forecast_df["temp"].iloc[t])
            c_dhw = self.cop_dhw.get_cop(forecast_df["temp"].iloc[t])

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
            constraints += [t_room[t] + slack_room_low[t] >= 19.0]
            constraints += [t_dhw[t] + slack_dhw_low[t] >= self.dhw_min]

            # Harde grenzen (Veiligheid)
            constraints += [t_room[t] <= 21.0, t_dhw[t] <= 55.0]

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

        solar_utilized = cp.minimum(
            p_el_ufh + p_el_dhw, forecast_df["power_corrected"].values
        )
        solar_bonus = cp.sum(solar_utilized) * 0.22  # 22 cent bonus per 'gratis' kWh

        # Anti-pendel: Straf het omschakelen of aan/uit gaan (MILP switches)
        switches = cp.sum(cp.abs(u_ufh[1:] - u_ufh[:-1])) + cp.sum(
            cp.abs(u_dhw[1:] - u_dhw[:-1])
        )

        # Comfort: probeer de kamer op 20.0 graden te houden
        comfort_tracking = cp.sum(cp.abs(t_room - 20.0)) * 0.05

        # Slack boete: Zeer hoog om comfortgrenzen te bewaken
        violation_penalty = cp.sum(slack_room_low + slack_dhw_low) * 25.0

        dhw_comfort = cp.abs(t_dhw[T] - self.dhw_target) * 5.0

        # Totaal te minimaliseren
        objective = cp.Minimize(
            cost  # Minimaliseer rekening van de leverancier
            - solar_bonus  # Maximaliseer gebruik van eigen zon
            + 1.0 * switches  # Voorkom pendelen (anti-slijtage)
            + comfort_tracking  # Houd de kamer op 20 graden
            + violation_penalty  # Nooit onder de 35 graden in de boiler
            + dhw_comfort  # Houd de boiler rond de target aan het einde
        )

        # 4. SOLVE
        problem = cp.Problem(objective, constraints)

        try:
            # CBC via CyLP is de aanbevolen MILP solver
            problem.solve(solver=cp.CBC, verbose=True)
        except Exception as e:
            logger.warning(
                f"[Optimizer] CBC solver niet beschikbaar, probeer andere MILP solvers. Fout: {e}"
            )
            # Fallback naar andere beschikbare MILP solvers (GLPK, SCIP)
            problem.solve(verbose=True)

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
            efficiency=0.50, cop_min=2.5, cop_max=5.0
        )
        self.cop_dhw = COPModel(
            efficiency=0.45, cop_min=1.5, cop_max=3.0
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

        logger.debug(f"[Optimizer] UFH residuals: {ufh_residuals}")
        logger.debug(f"[Optimizer] DHW residuals: {dhw_residuals}")

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
