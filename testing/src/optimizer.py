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
# 1. SYSTEEM IDENTIFICATIE (RC, COP-vrij)
# =========================================================
class SystemIdentificator:
    """
    Identificeert R en C
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
        df = df.set_index("timestamp").sort_index()
        # Resample naar 15min en vul kleine gaatjes op om gaten in de regressie te voorkomen
        df = df.resample("15min").interpolate(method='linear', limit=2)

        # 1. Bepaal de verandering over de KOMENDE 60 minuten (4 stappen van 15m)
        df["dT_1h"] = df["room_temp"].shift(-4) - df["room_temp"]

        # 2. Gebruik het nieuwe COP model
        # We gaan uit van een gemiddelde efficiëntie voor vloerverwarming
        cop_calc = COPModel(efficiency=0.50)

        # Bereken de COP per tijdstap op basis van de werkelijke metingen.
        # Sink temperatuur voor UFH is meestal kamer_temp + 5 graden aanvoer.
        df["cop_actual"] = df.apply(
            lambda row: cop_calc.get_cop(row["temp"], row["room_temp"] + 5.0),
            axis=1
        )

        # Bereken het werkelijke thermische vermogen dat de woning in ging
        df["p_th_actual"] = df["wp_actual"] * df["cop_actual"]

        # 3. Features voorbereiden (gemiddelden over de komende 4 kwartieren)
        # We verschuiven deze met shift(-4) zodat X1/X2 over hetzelfde uur gaan als dT_1h

        # X1: Gemiddeld temperatuurverschil (verlies naar buiten)
        loss_potential = -(df["room_temp"] - df["temp"])
        df["X1"] = loss_potential.rolling(window=4).mean().shift(-3)

        # X2: Gemiddeld thermisch vermogen (winst van de warmtepomp)
        df["X2"] = df["p_th_actual"].rolling(window=4).mean().shift(-3)

        # 4. Filteren voor een zuivere identificatie
        # - Geen zon (pv_actual < 0.05): zoninstraling verstoort de R-waarde berekening enorm.
        # - Stabiele dT: voorkom meetfouten of open ramen/deuren.
        mask = (df["pv_actual"] < 0.05) & (df["dT_1h"].abs() < 1.5)
        train_data = df[mask][["X1", "X2", "dT_1h"]].dropna()

        if len(train_data) < 50:
            logger.info("[Optimizer] Te weinig stabiele data (zonder zon) voor RC identificatie.")
            return

        X = train_data[["X1", "X2"]]
        y = train_data["dT_1h"]

        # 5. Regressie: y = (1/C) * X2 + (1/RC) * X1  (ongeveer)
        # Eigenlijk: dT = (P_th / C) - ( (T_room - T_out) / (R*C) )
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        coef_loss, coef_gain = model.coef_

        # 6. Fysieke waarden afleiden met begrenzingen (clamping)
        # Gain coefficient = 1 / C
        c_gain_clamped = np.clip(coef_gain, 1.0 / 120.0, 1.0 / 15.0)
        self.C = 1.0 / c_gain_clamped

        # Loss coefficient = 1 / (R * C)
        # Dus R = 1 / (coef_loss * C)
        r_loss_clamped = np.clip(coef_loss, 1.0 / (40.0 * self.C), 1.0 / (5.0 * self.C))
        self.R = 1.0 / (r_loss_clamped * self.C)

        self.is_fitted = True
        joblib.dump({"R": self.R, "C": self.C}, self.path)
        logger.info(
            f"[Identificatie] R={self.R:.2f} K/kW, C={self.C:.2f} kWh/K"
        )

# =========================================================
# 2. COP-MODEL
# =========================================================
class COPModel:
    def __init__(self, efficiency=0.45, cop_min=1.2, cop_max=6.0):
        self.eta = efficiency
        self.cop_min = cop_min
        self.cop_max = cop_max

    def get_cop(self, T_out, T_sink):
        """
        T_sink is de temperatuur van het water (bijv. 30C voor UFH of 45C voor DHW).
        """
        # Voeg 3 graden toe aan T_sink voor de warmtewisselaar delta
        T_sink_k = T_sink + 273.15 + 3.0
        T_out_k = T_out + 273.15

        # Carnot limiet: T_sink / (T_sink - T_out)
        delta_t = np.maximum(T_sink_k - T_out_k, 5.0) # Minimaal 5 graden verschil
        cop_theoretical = T_sink_k / delta_t

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
        df = df.set_index("timestamp").sort_index()
        # Gebruik interpolatie voor kleine gaatjes in de data
        df = df.resample("15min").interpolate(method='linear', limit=2)
        dt = 0.25
        boiler_mass_factor = 0.232

        # Voeg cyclische tijd features toe
        df = df.reset_index()
        df = add_cyclic_time_features(df, col_name="timestamp")

        # 1. BEREKEN DE COP PER RIJ
        # De COP hangt nu af van buiten_temp en de sink_temp
        if not is_dhw:
            # UFH: Sink is kamer_temp + 7 graden aanvoer-delta
            df["cop_calc"] = df.apply(
                lambda row: cop_model.get_cop(row["temp"], row["room_temp"] + 5.0),
                axis=1
            )

            # Theoretische delta (RC-model)
            # dT = ((P_el * COP) - (T_room - T_out) / R) * dt / C
            loss = (df["room_temp"] - df["temp"]) / R
            dT_rc = ((df["wp_actual"] * df["cop_calc"]) - loss) * dt / C

            # Werkelijke delta
            y_actual = df["room_temp"].shift(-1) - df["room_temp"]

            df["solar"] = df["pv_actual"] # Nodig voor ufh features
            feature_cols = self.features_ufh
        else:
            # DHW: Sink is de temperatuur van het water
            if "dhw_temp" not in df.columns:
                df["dhw_temp"] = df["dhw_top"] * 0.7 + df["dhw_bottom"] * 0.3

            df["cop_calc"] = df.apply(
                lambda row: cop_model.get_cop(row["temp"], row["dhw_temp"] + 9.0),
                axis=1
            )

            # Theoretische winst in de boiler (zonder verliezen)
            dT_rc = (df["wp_actual"] * df["cop_calc"] * dt) / boiler_mass_factor

            # Werkelijke delta
            y_actual = df["dhw_temp"].shift(-1) - df["dhw_temp"]
            feature_cols = self.features_dhw

        # 2. BEREKEN RESIDUAL
        # Residual = Werkelijkheid - Theorie
        # Dit is wat het ML model moet leren voorspellen (bijv. zoninstraling of douche-verbruik)
        target_series = (y_actual - dT_rc).rename("target_residual")

        if is_dhw:
            # Sta het model toe om dalingen tot 1.5 graad per kwartier te 'leren'
            target_series = target_series.clip(lower=-1.5, upper=0.2)

        # 3. TRAINING
        X = df.reindex(columns=feature_cols)
        train_df = pd.concat([X, target_series], axis=1).dropna()

        if len(train_df) > 50:
            X_train = train_df[feature_cols]
            y_train = train_df["target_residual"]

            self.model = HistGradientBoostingRegressor()
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.path)
            logger.info(
                f"[Optimizer] ML Model {self.path.name} getraind op {len(X_train)} samples."
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

        preds = self.model.predict(X_predict)

        if is_dhw:
            # BOILER: Moet douche-beurten herkennen, maar beperkt tot 'gecontroleerde' daling
            # -0.5 is de 'sweet spot' voor routineherkenning zonder paniekstoken.
            return np.clip(preds, -1.5, 0.05)
        else:
            # WOONKAMER: Moet heel stabiel zijn.
            # Correcties van meer dan 0.1 graad per kwartier zijn vaak ruis.
            # Dit houdt het stookgedrag voor de vloer heel rustig.
            return np.clip(preds, -0.1, 0.1)


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
        Geadvanceerde MPC solve functie met zon-prioriteit en douche-planning.
        """
        T = self.horizon
        dt = 0.25  # Kwartier
        R, C = self.ident.R, self.ident.C
        boiler_mass_factor = 0.232
        dhw_standby_loss = 0.04

        # 1. VARIABELEN
        u_ufh = cp.Variable(T, boolean=True)
        u_dhw = cp.Variable(T, boolean=True)
        p_el_ufh = cp.Variable(T, nonneg=True)
        p_el_dhw = cp.Variable(T, nonneg=True)

        t_room = cp.Variable(T + 1)
        t_dhw = cp.Variable(T + 1)

        # Slack variabelen voor gelaagd comfort
        slack_room_low = cp.Variable(T, nonneg=True)     # Harde grens 19.0
        slack_room_comfort = cp.Variable(T, nonneg=True) # Comfort grens 19.6 (wachten op zon)
        slack_dhw_low = cp.Variable(T, nonneg=True)      # Harde grens 35.0
        slack_dhw_high = cp.Variable(T, nonneg=True)     # Overschrijding 50.0 (alleen bij zon)

        dhw_start_temp = (state["dhw_top"] * 0.7 + state["dhw_bottom"] * 0.3)

        # 2. CONSTRAINTS
        constraints = [
            t_room[0] == state["room_temp"],
            t_dhw[0] == dhw_start_temp,
        ]

        # Schattingen voor lineaire COP berekening
        estimated_t_dhw = dhw_start_temp
        estimated_t_room = state["room_temp"]

        for t in range(T):
            # A. COP Bepaling (Physics Based Heuristic)
            t_sink_ufh = estimated_t_room + 5.0
            t_sink_dhw = estimated_t_dhw + 9.0

            c_ufh_t = self.cop_ufh.get_cop(forecast_df["temp"].iloc[t], t_sink_ufh)
            c_dhw_t = self.cop_dhw.get_cop(forecast_df["temp"].iloc[t], t_sink_dhw)

            # B. WP Systeem Constraints
            constraints += [u_ufh[t] + u_dhw[t] <= 1] # Exclusiviteit
            constraints += [p_el_dhw[t] == self.p_el_max * u_dhw[t]] # Boiler altijd vol vermogen

            # UFH vermogen met koude-correctie
            temp_correction = cp.maximum(0, (0 - forecast_df["temp"].iloc[t]) * 0.05)
            min_p_ufh = (0.8 + temp_correction) * u_ufh[t]

            constraints += [p_el_ufh[t] >= min_p_ufh]
            constraints += [p_el_ufh[t] <= self.p_el_max * u_ufh[t]]

            # C. Dynamica (RC-model + Boiler massa + Residuals)
            constraints += [
                t_room[t+1] == t_room[t] +
                ((p_el_ufh[t] * c_ufh_t) - (t_room[t] - forecast_df["temp"].iloc[t])/R) * dt/C
                + ufh_residuals[t]
            ]

            constraints += [
                t_dhw[t+1] == t_dhw[t] +
                (p_el_dhw[t] * c_dhw_t * dt) / boiler_mass_factor
                - dhw_standby_loss + dhw_residuals[t]
            ]

            # D. Comfort & Veiligheid Constraints (met Slacks)
            constraints += [t_room[t+1] + slack_room_low[t] >= 19.0]     # Nooit onder 19
            constraints += [t_room[t+1] + slack_room_comfort[t] >= 19.4] # Wacht op zon
            constraints += [t_dhw[t+1] + slack_dhw_low[t] >= self.dhw_min] # Nooit onder 35

            # De 50 graden 'zachte' bovengrens
            constraints += [t_dhw[t+1] <= 50.0 + slack_dhw_high[t]]

            # Harde fysieke grenzen
            constraints += [t_room[t+1] <= 22.0, t_dhw[t+1] <= 60.0]

            # E. Update schatting voor volgende tijdstap
            # Gebruik 70% van max vermogen als aanname voor stijging
            rise_dhw = (3.0 * c_dhw_t * dt) / boiler_mass_factor
            estimated_t_dhw = min(estimated_t_dhw + rise_dhw, 60.0)

            rise_room = ((1.5 * c_ufh_t) - (estimated_t_room - forecast_df["temp"].iloc[t])/R) * dt/C
            estimated_t_room = min(estimated_t_room + max(0, rise_room), 22.0)

        # 3. OBJECTIVE (Kosten + Boetes)
        # Netto import
        net_load = (p_el_ufh + p_el_dhw +
                    forecast_df["load_corrected"].values -
                    forecast_df["power_corrected"].values)

        cost = cp.sum(cp.multiply(cp.pos(net_load), prices)) * dt

        # Zonne-energie bonus (0.22 Euro per kWh eigen verbruik)
        solar_utilized = cp.minimum(p_el_ufh + p_el_dhw, forecast_df["power_corrected"].values)
        solar_bonus = cp.sum(solar_utilized) * 0.22

        # Schakelkosten (Anti-pendel)
        switches = cp.sum(cp.abs(u_ufh[1:] - u_ufh[:-1])) + cp.sum(cp.abs(u_dhw[1:] - u_dhw[:-1]))

        # --- KAMER LOGICA ---
        # 1. Onder de 19.6: Hoge boete (0.6), dwingt stoken met netstroom af
        room_comfort_penalty = cp.sum(slack_room_comfort) * 0.6
        # 2. Tussen 19.6 en 20.0: Lage boete (0.05), wacht op zon of daltarief
        room_target_tracking = cp.sum(cp.pos(20.0 - t_room)) * 0.05

        # --- BOILER LOGICA ---
        # 1. Onder de 50.0: Boete (0.1), dwingt vulling af voor douche, maar wacht na douche op zon
        dhw_under_target = cp.sum(cp.pos(50.0 - t_dhw)) * 0.1
        # 2. Boven de 50.0: Kleine boete (0.02), alleen rendabel bij zon (0.22)
        dhw_overheat_penalty = cp.sum(slack_dhw_high) * 0.02

        # 3. Harde grens boete (Veiligheid)
        violation_penalty = cp.sum(slack_room_low + slack_dhw_low) * 25.0

        # TOTAAL
        objective = cp.Minimize(
            cost
            - solar_bonus
            + (1.0 * switches)
            + room_comfort_penalty
            + room_target_tracking
            + dhw_under_target
            + dhw_overheat_penalty
            + violation_penalty
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

        self.cop_ufh = COPModel(efficiency=0.50, cop_min=2.5, cop_max=6.0)
        self.cop_dhw = COPModel(efficiency=0.45, cop_min=1.5, cop_max=4.5)

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
