import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging

from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from config import Config
from context import Context
from database import Database
from utils import add_cyclic_time_features

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
        # Aanname: Vloerverwarming water is kamer_temp + 5 graden
        cop_calc = COPModel(efficiency=0.50)
        df["cop_actual"] = df.apply(
            lambda row: cop_calc.get_cop(row["temp"], row["room_temp"] + 5.0), axis=1
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
# 4. MPC (Optimization Core)
# =========================================================
class ThermalMPC:
    def __init__(
        self, ident: SystemIdentificator, cop_ufh: COPModel, cop_dhw: COPModel
    ):
        self.ident = ident
        self.cop_ufh_model = cop_ufh
        self.cop_dhw_model = cop_dhw

        # Instellingen
        self.horizon = 48  # 12 uur
        self.dt = 0.25  # 15 min
        self.p_el_max = 3.5  # kW
        self.dhw_min = 30.0  # Absolute ondergrens
        self.dhw_target = 50.0
        self.cold_water = 15.0

        # Bouw de wiskundige structuur (DPP)
        self._build_problem()

    def _build_problem(self):
        """Constructs the CVXPY problem structure using Parameters for speed."""
        T = self.horizon

        # --- PARAMETERS (Inputs) ---
        self.P_t_room_init = cp.Parameter(name="t_room_init")
        self.P_t_dhw_init = cp.Parameter(name="t_dhw_init")

        self.P_temp_out = cp.Parameter(T, name="temp_out")
        self.P_prices = cp.Parameter(T, nonneg=True, name="prices")
        self.P_solar_avail = cp.Parameter(T, nonneg=True, name="solar")
        self.P_load_base = cp.Parameter(T, name="load_base")

        self.P_ufh_res = cp.Parameter(T, name="ufh_res")
        self.P_dhw_res = cp.Parameter(T, name="dhw_res")

        # Voorberekende vectoren (Pythonside logic)
        self.P_cop_ufh = cp.Parameter(T, nonneg=True, name="cop_ufh")
        self.P_cop_dhw = cp.Parameter(T, nonneg=True, name="cop_dhw")
        self.P_min_p_ufh = cp.Parameter(T, nonneg=True, name="min_p_ufh")

        # --- VARIABELEN ---
        self.u_ufh = cp.Variable(T, boolean=True, name="u_ufh")
        self.u_dhw = cp.Variable(T, boolean=True, name="u_dhw")

        self.p_el_ufh = cp.Variable(T, nonneg=True, name="p_el_ufh")
        self.p_el_dhw = cp.Variable(T, nonneg=True, name="p_el_dhw")

        # NIEUW: Hulpvariabele voor grid import (DPP Fix)
        self.p_grid_import = cp.Variable(T, nonneg=True, name="p_grid_import")

        # NIEUW: Hulpvariabele voor Solar Self-Consumption (vermijdt cp.minimum)
        self.p_solar_self = cp.Variable(T, nonneg=True, name="p_solar_self")

        self.t_room = cp.Variable(T + 1, name="t_room")
        self.t_dhw = cp.Variable(T + 1, name="t_dhw")

        # Slack variabelen (Soft Constraints)
        self.slack_room_low = cp.Variable(T, nonneg=True)  # < 19.0
        self.slack_room_comfort = cp.Variable(T, nonneg=True)  # < 19.4 (zon buffer)
        self.slack_room_high = cp.Variable(T, nonneg=True)  # > 22.0 (overheat)

        self.slack_dhw_low = cp.Variable(T, nonneg=True)  # < 35.0
        self.slack_dhw_high = cp.Variable(T, nonneg=True)  # > 50.0 (zon buffer)

        # --- CONSTANTEN ---
        R, C = self.ident.R, self.ident.C
        boiler_mass_factor = 0.232  # 200L
        dhw_standby_loss = 0.04

        # --- CONSTRAINTS ---
        constraints = [
            self.t_room[0] == self.P_t_room_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        for t in range(T):
            # 1. Hardware Limits
            constraints += [
                self.u_ufh[t] + self.u_dhw[t] <= 1,
                # DHW: Altijd max vermogen voor efficiëntie
                self.p_el_dhw[t] == self.p_el_max * self.u_dhw[t],
                # UFH: Modulerend tussen min en max
                self.p_el_ufh[t] <= self.p_el_max * self.u_ufh[t],
                self.p_el_ufh[t] >= self.P_min_p_ufh[t] * self.u_ufh[t],
            ]

            # 2. Thermische Dynamica (Room)
            # Winst = P_el * COP. Verlies = DeltaT / R.
            p_th_room = self.p_el_ufh[t] * self.P_cop_ufh[t]
            loss_room = (self.t_room[t] - self.P_temp_out[t]) / R

            constraints += [
                self.t_room[t + 1]
                == self.t_room[t]
                + (p_th_room - loss_room) * self.dt / C
                + self.P_ufh_res[t]
            ]

            # 3. Thermische Dynamica (DHW)
            p_th_dhw = self.p_el_dhw[t] * self.P_cop_dhw[t]

            constraints += [
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (p_th_dhw * self.dt) / boiler_mass_factor
                - dhw_standby_loss
                + self.P_dhw_res[t],
            ]

            # 4. Comfort & Veiligheid
            constraints += [
                # Ondergrens
                self.t_room[t + 1] + self.slack_room_low[t] >= 19.0,
                self.t_room[t + 1] + self.slack_room_comfort[t] >= 19.4,
                self.t_dhw[t + 1] + self.slack_dhw_low[t] >= self.dhw_min,
                # Bovengrens (Soft)
                self.t_dhw[t + 1] <= 50.0 + self.slack_dhw_high[t],
                self.t_room[t + 1] <= 22.0 + self.slack_room_high[t],
                self.t_dhw[t + 1] <= 65.0,
                self.t_room[t + 1] <= 25.0,
                # Supply Temp Proxy (Vloer bescherming, max 30C)
                self.t_room[t] + self.p_el_ufh[t] * 4.0 <= 30,
            ]

            # 4. Net Load & Solar Logic (Linearized)
            total_load = self.p_el_ufh[t] + self.p_el_dhw[t]
            net_load_expr = total_load + self.P_load_base[t] - self.P_solar_avail[t]

            constraints += [
                # Import is wat we te kort komen
                self.p_grid_import[t] >= net_load_expr,
                # Solar Self Consumption Logic:
                # p_solar_self mag niet meer zijn dan de Load
                self.p_solar_self[t] <= total_load,
                # p_solar_self mag niet meer zijn dan de Beschikbare Zon
                self.p_solar_self[t] <= self.P_solar_avail[t],
                # Door deze variabele te maximaliseren in de objective,
                # zoekt de solver de 'overlap' tussen Load en Zon.
            ]

        # --- OBJECTIVE ---
        # Gebruik de hulpvariabele p_grid_import i.p.v. cp.pos(net_load)
        # Dit is nu: Variabele * Parameter. Dat is DPP compliant.
        cost = cp.sum(cp.multiply(self.p_grid_import, self.P_prices)) * self.dt

        # Solar Bonus (Nu lineair en snel!)
        solar_bonus = cp.sum(self.p_solar_self) * 0.22

        # Anti-Pendel Fix: Start-up penalty
        # We straffen (u[t] - u[t-1]) alleen als het positief is.
        # Dit is DCP compliant omdat cp.pos convex is en we minimaliseren.
        # We gebruiken slicing [1:] en [:-1] voor vectoren.
        startups_ufh = cp.sum(cp.pos(self.u_ufh[1:] - self.u_ufh[:-1]))
        startups_dhw = cp.sum(cp.pos(self.u_dhw[1:] - self.u_dhw[:-1]))
        switches = (startups_ufh + startups_dhw) * 2.0  # Straf factor

        # Comfort Penalties
        # 1. Te koud (Hard & Comfort)
        pen_room_cold = (
            cp.sum(self.slack_room_low) * 25.0 + cp.sum(self.slack_room_comfort) * 0.6
        )
        pen_dhw_cold = cp.sum(self.slack_dhw_low) * 25.0

        # 2. Te warm (Soft)
        pen_room_hot = cp.sum(self.slack_room_high) * 10.0
        pen_dhw_hot = cp.sum(self.slack_dhw_high) * 0.02

        # 3. Target Tracking (Kamer 20C, DHW 50C)
        track_room = cp.sum(cp.pos(20.0 - self.t_room)) * 0.05
        track_dhw = cp.sum(cp.pos(50.0 - self.t_dhw)) * 0.1

        total_obj = (
            cost
            - solar_bonus
            + switches
            + pen_room_cold
            + pen_dhw_cold
            + pen_room_hot
            + pen_dhw_hot
            + track_room
            + track_dhw
        )

        self.problem = cp.Problem(cp.Minimize(total_obj), constraints)

        # Deze assert zou nu moeten slagen
        assert (
            self.problem.is_dpp()
        ), "Probleem is niet DPP compliant! Check parameters."

    def solve(self, state, forecast_df, prices, ufh_residuals, dhw_residuals):
        # 1. BEREKEN VECTOREN (Python Logic Simulation)
        # We simuleren hier de temperatuurverloop om de COP te schatten
        temps = forecast_df["temp"].values
        T = self.horizon
        dt = self.dt

        # Init waarden
        dhw_start = (state["dhw_top"] + state["dhw_bottom"]) / 2
        sim_t_dhw = dhw_start
        sim_t_room = state["room_temp"]

        v_cop_ufh = []
        v_cop_dhw = []
        v_min_p_ufh = []

        for t in range(T):
            temp_out = temps[t]

            # --- UFH Logic ---
            t_sink_ufh = sim_t_room + 6.0
            cop_u = self.cop_ufh_model.get_cop(temp_out, t_sink_ufh)
            v_cop_ufh.append(cop_u)

            v_min_p_ufh.append(0.8 + max(0, (0 - temp_out) * 0.05))

            # om de temperatuurstijging voor de COP-schatting te simuleren.
            p_el_est_ufh = self.p_el_max * 0.4
            rise_room = (
                ((p_el_est_ufh * cop_u) - (sim_t_room - temp_out) / self.ident.R)
                * dt
                / self.ident.C
            )
            sim_t_room = np.clip(sim_t_room + rise_room, 18.0, 22.0)

            # --- DHW Logic ---
            t_sink_dhw = sim_t_dhw + 9.0
            cop_d = self.cop_dhw_model.get_cop(temp_out, t_sink_dhw)
            v_cop_dhw.append(cop_d)

            # Maar beperk de thermische winst tot de fysieke limiet (bijv. 7.0 kW)
            p_th_est_dhw = min(self.p_el_max * cop_d, 7.0)
            rise_dhw = (p_th_est_dhw * dt) / 0.232
            sim_t_dhw = np.clip(sim_t_dhw + rise_dhw, 15.0, 60.0)

        # 2. PARAMETERS VULLEN
        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = dhw_start

        self.P_temp_out.value = temps
        self.P_prices.value = np.array(prices)
        self.P_solar_avail.value = forecast_df["power_corrected"].values
        self.P_load_base.value = forecast_df["load_corrected"].values

        self.P_ufh_res.value = np.array(ufh_residuals)
        self.P_dhw_res.value = np.array(dhw_residuals)

        self.P_cop_ufh.value = np.array(v_cop_ufh)
        self.P_cop_dhw.value = np.array(v_cop_dhw)
        self.P_min_p_ufh.value = np.array(v_min_p_ufh)

        # 3. SOLVE
        try:
            # CBC is de beste open-source MILP solver
            self.problem.solve(
                solver=cp.CBC, verbose=True, maximumSeconds=60, allowableGap=0.05
            )
        except Exception:
            logger.warning("[Optimizer] CBC faalde, fallback naar GLPK_MI")
            try:
                self.problem.solve(solver=cp.GLPK_MI, verbose=True)
            except Exception as e:
                logger.error(f"[Optimizer] Solver Critical Error: {e}")
                return None

        if self.u_ufh.value is None:
            logger.error(f"[Optimizer] Infeasible. Status: {self.problem.status}")
            return None

        # 4. FORMATTEER RESULTATEN
        mode = (
            "DHW"
            if self.u_dhw.value[0] > 0.5
            else "UFH" if self.u_ufh.value[0] > 0.5 else "OFF"
        )
        current_p_el = float(self.p_el_ufh.value[0] + self.p_el_dhw.value[0])

        dhw_avg = (state["dhw_top"] + state["dhw_bottom"]) / 2
        soc = (dhw_avg - self.cold_water) / (self.dhw_target - self.cold_water)
        dhw_kwh = max(0.0, (dhw_avg - self.cold_water) * 0.232)

        # Dit is puur: Import * Prijs
        real_cost_import = (
            np.sum(self.p_grid_import.value * self.P_prices.value) * self.dt
        )

        return {
            "mode": mode,
            "target_power": round(current_p_el, 3),
            "planned_room": self.t_room.value.tolist(),
            "planned_dhw": self.t_dhw.value.tolist(),
            "dhw_soc": max(0, min(1, soc)),
            "dhw_energy_kwh": dhw_kwh,
            "cost_projected": round(real_cost_import, 2),
            "objective_score": float(self.problem.value),
            "status": self.problem.status,
        }


# =========================================================
# 5. EMS WRAPPER
# =========================================================
class Optimizer:
    def __init__(self, config: Config, database: Database):
        self.database = database
        self.ident = SystemIdentificator(config.rc_model_path)

        # UFH: Efficiëntie ~0.50 (incl pompen), min COP 2.5
        self.cop_ufh = COPModel(efficiency=0.50, cop_min=2.5, cop_max=6.0)
        # DHW: Iets lager rendement door hogere temperatuur eisen
        self.cop_dhw = COPModel(efficiency=0.45, cop_min=1.5, cop_max=4.5)

        self.ufh_res = MLResidualPredictor(config.ufh_model_path)
        self.dhw_res = MLResidualPredictor(config.dhw_model_path)

        self.mpc = ThermalMPC(self.ident, self.cop_ufh, self.cop_dhw)

    def resolve(self, context: Context):
        # Pak de horizon uit de context
        horizon_df = context.forecast_df.iloc[: self.mpc.horizon].copy()

        # Voorspel verstoringen (Zon door ramen, Douchen)
        ufh_residuals = self.ufh_res.predict(horizon_df, is_dhw=False)
        dhw_residuals = self.dhw_res.predict(horizon_df, is_dhw=True)

        state = {
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        # Dynamische prijzen (indien beschikbaar in context, anders dummy)
        prices = (
            context.prices if hasattr(context, "prices") else [0.21] * len(horizon_df)
        )
        if len(prices) < len(horizon_df):
            prices = [0.21] * len(horizon_df)

        return self.mpc.solve(state, horizon_df, prices, ufh_residuals, dhw_residuals)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)
        history_df = self.database.get_history(cutoff_date=cutoff)

        if history_df.empty:
            logger.warning("[Optimizer] Geen data om te trainen.")
            return

        logger.info("[Optimizer] Start training System ID...")
        self.ident.train(history_df)

        logger.info("[Optimizer] Start training ML Residuals...")
        self.ufh_res.train(
            history_df, self.ident.R, self.ident.C, self.cop_ufh, is_dhw=False
        )
        self.dhw_res.train(
            history_df, self.ident.R, self.ident.C, self.cop_dhw, is_dhw=True
        )
