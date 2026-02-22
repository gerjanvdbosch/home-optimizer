import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging

from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils import add_cyclic_time_features
from context import Context, HvacMode

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================
# Deze constanten worden nu gebruikt als fallback of voor initiële schattingen
FLOW_UFH = 18.0  # Liter per minuut
FLOW_DHW = 19.0  # Liter per minuut
CP_WATER = 4.186  # kJ/kg.K

# Bereken kW per Kelvin delta_T: (Flow / 60) * Cp
FACTOR_UFH = (FLOW_UFH / 60.0) * CP_WATER  # ~1.2558 kW/K
FACTOR_DHW = (FLOW_DHW / 60.0) * CP_WATER  # ~1.3256 kW/K


# =========================================================
# 1. HEAT PUMP PERFORMANCE MAP
# =========================================================
class HPPerformanceMap:
    def __init__(self, path):
        self.path = Path(path)
        self.is_fitted = False
        self.cop_model = None

        # Modellen om de grenzen van de warmtepomp te leren
        self.max_pel_model = None
        self.min_pel_model = None

        # Features vereenvoudigd: Geen frequentie of supply temps meer nodig!
        # De ML leert de verwachte COP puur op basis van de bron (buiten) en afgifte (kamer/boiler)
        self.features_cop = ["temp", "sink_temp", "hvac_mode"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.cop_model = data["cop_model"]
                self.max_pel_model = data.get("max_pel_model")
                self.min_pel_model = data.get("min_pel_model")
                self.is_fitted = True
                logger.info("[Optimizer] Performance map (Zonder frequentie) geladen.")
            except Exception as e:
                logger.warning(f"[Optimizer] Performance map laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy()

        # ============================
        # 1. Fysisch thermisch vermogen berekenen
        # ============================
        df["delta_t"] = (df["supply_temp"] - df["return_temp"]).astype(float)
        df = df[df["delta_t"] > 0.5]

        conditions = [
            df["hvac_mode"] == HvacMode.HEATING.value,
            df["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices = [
            df["delta_t"] * FACTOR_UFH, # Hier gebruiken we nog de vaste factor voor historische data verwerking
            df["delta_t"] * FACTOR_DHW,
        ]
        df["wp_output"] = np.select(conditions, choices, default=0.0)

        # Bepaal de 'sink' temperatuur (waar gaat de warmte heen?)
        choices_sink = [df["room_temp"], df["dhw_bottom"]]
        df["sink_temp"] = np.select(conditions, choices_sink, default=20.0)

        df = df[(df["wp_output"] > 0.1) & (df["wp_actual"] > 0.1)].copy()
        df["cop"] = df["wp_output"] / df["wp_actual"]

        mask = (df["cop"].between(0.8, 8.0)) & (df["delta_t"].between(1.0, 15.0))
        df_clean = df[mask].copy()

        if len(df_clean) < 50:
            logger.warning("[Optimizer] Te weinig valide data voor performance map training.")
            return

        # ============================
        # 2. COP model (Random Forest)
        # ============================
        self.cop_model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42)
        self.cop_model.fit(df_clean[self.features_cop], df_clean["cop"])

        # ============================
        # 3. Leer de grenzen van de warmtepomp (Elektrisch vermogen)
        # ============================
        df_f = df_clean.copy()
        df_f["t_rounded"] = df_f["temp"].round()

        # Wat is het maximale en minimale elektrische vermogen per buitentemperatuur?
        max_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.98).reset_index()
        min_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.05).reset_index()

        if not max_stats.empty:
            self.max_pel_model = LinearRegression().fit(max_stats[["t_rounded"]], max_stats["wp_actual"])
            self.min_pel_model = LinearRegression().fit(min_stats[["t_rounded"]], min_stats["wp_actual"])

        self.is_fitted = True
        joblib.dump({
            "cop_model": self.cop_model,
            "max_pel_model": self.max_pel_model,
            "min_pel_model": self.min_pel_model
        }, self.path)
        logger.info("[Optimizer] Performance Map getraind (Machine Learning over fysica).")

    def predict_cop(self, t_out, t_sink, mode_idx):
        if not self.is_fitted:
            return 3.5 if mode_idx == HvacMode.HEATING.value else 2.5
        data = pd.DataFrame([[t_out, t_sink, mode_idx]], columns=self.features_cop)
        return float(self.cop_model.predict(data)[0])

    def get_pel_limits(self, t_out):
        """Geeft (min_kW, max_kW) terug op basis van wat de machine historisch doet."""
        if not self.is_fitted or self.max_pel_model is None:
            return 0.4, 2.5  # Veilig default: 400W tot 2500W

        t_df = pd.DataFrame([[round(t_out)]], columns=["t_rounded"])
        min_p = float(self.min_pel_model.predict(t_df)[0])
        max_p = float(self.max_pel_model.predict(t_df)[0])

        # Zorg voor logische grenzen
        return np.clip(min_p, 0.2, 1.0), np.clip(max_p, 1.5, 4.0)


# =========================================================
# 2. SYSTEM IDENTIFICATOR (MET TRAAGHEID)
# =========================================================
class SystemIdentificator:
    def __init__(self, path):
        self.path = Path(path)
        self.R = 15.0  # K/kW
        self.C = 30.0  # kWh/K
        self.K_emit = 0.15  # kW/K (Afgifte vloer)
        self.K_tank = 0.25  # kW/K (Afgifte spiraal)
        self.K_loss_dhw = 0.01  # °C/uur verlies
        self.ufh_lag_steps = 4  # Aantal 15-minuten stappen traagheid (default 1 uur)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.R = data.get("R", 15.0)
                self.C = data.get("C", 30.0)
                self.K_emit = data.get("K_emit", 0.15)
                self.K_tank = data.get("K_tank", 0.25)
                self.K_loss_dhw = data.get("K_loss_dhw", 0.01)
                self.ufh_lag_steps = data.get("ufh_lag_steps", 4)
                logger.info(
                    f"[SysID] Geladen: Lag={self.ufh_lag_steps * 15}m, K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f} R={self.R:.1f} C={self.C:.1f} K_loss_dhw={self.K_loss_dhw:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to load SystemID: {e}")

    def train(self, df: pd.DataFrame):
        df_proc = df.copy().sort_values("timestamp")

        # Zorg dat alle relevante kolommen numeriek zijn
        for col in ["supply_temp", "return_temp", "room_temp", "temp", "hvac_mode", "pv_actual"]:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

        # Bereken thermisch vermogen
        df_proc["delta_t"] = (df_proc["supply_temp"] - df_proc["return_temp"]).clip(lower=0.0)
        df_proc["wp_output"] = np.where(df_proc["hvac_mode"] == HvacMode.HEATING.value,
                                        df_proc["delta_t"] * FACTOR_UFH, 0.0)

        # Resample naar 1 uur voor stabiele thermische modellering
        df_1h = df_proc.set_index("timestamp").resample("1h").mean().dropna(
            subset=['room_temp', 'temp', 'pv_actual', 'wp_output']).reset_index()

        df_1h["dT_next"] = df_1h["room_temp"].shift(-1) - df_1h["room_temp"]
        df_1h["delta_T_env"] = df_1h["room_temp"] - df_1h["temp"]

        # ==========================================================
        # STRATEGIE 1: De Fysisch Gesplitste Methode (Voorkeur)
        # ==========================================================
        logger.info("[SysID] Poging 1: Fysisch gesplitste R/C training.")

        # --- STAP 1A: Afkoeling (Bepaal Tau = R * C) ---
        # Filter (iets minder streng gemaakt dan uw voorbeeld)
        mask_cooling = (
                (df_1h["wp_output"] < 0.1)
                & (df_1h["pv_actual"] < 0.05)  # Geen zoninvloed
                & (df_1h["dT_next"] < -0.01)  # Temperatuur daalt
                & (df_1h["delta_T_env"] > 3.0)  # Minimaal 3 graden kouder buiten
        )
        df_cool = df_1h[mask_cooling].copy()

        tau_inv = None  # Dit is 1 / (R * C)

        if len(df_cool) > 10:
            # Model: dT/dt = -1/(RC) * (T_room - T_out)  =>  Y = a * X
            X_cool = -df_cool[["delta_T_env"]]
            y_cool = df_cool["dT_next"]
            lr_cool = LinearRegression(fit_intercept=False).fit(X_cool, y_cool)

            # Controleer of de coëfficiënt een realistische waarde heeft
            if lr_cool.coef_[0] > 0:
                tau_inv = lr_cool.coef_[0]
                tau = 1 / tau_inv
                logger.info(
                    f"[SysID] Afkoelfase geïdentificeerd met {len(df_cool)} datapunten. Geschatte Tau (RC) = {tau:.1f} uur.")

                # Sanity Check: Tau moet binnen realistische grenzen vallen
                if not (20 < tau < 1000):
                    logger.warning(
                        f"[SysID] Tau waarde ({tau:.1f}) is onrealistisch. Fallback naar gecombineerde methode.")
                    tau_inv = None  # Reset om de fallback te triggeren
            else:
                logger.warning(
                    "[SysID] Negatieve coëfficiënt gevonden in afkoelfase, data is waarschijnlijk te ruizig.")

        else:
            logger.warning(
                f"[SysID] Te weinig data voor afkoelfase ({len(df_cool)} punten). Fallback naar gecombineerde methode.")

        # --- STAP 1B: Verwarming (Bepaal C, en dan R) ---
        if tau_inv is not None:
            mask_heat = (df_1h["wp_output"] > 0.5) & (df_1h["dT_next"] > 0.01)
            df_heat = df_1h[mask_heat].copy()

            if len(df_heat) > 10:
                y_heat_adjusted = df_heat["dT_next"] + (tau_inv * df_heat["delta_T_env"])
                X_heat = df_heat[["wp_output"]]
                lr_heat = LinearRegression(fit_intercept=False).fit(X_heat, y_heat_adjusted)

                if lr_heat.coef_[0] > 0:
                    new_C = 1.0 / lr_heat.coef_[0]
                    new_R = (1.0 / tau_inv) / new_C

                    if 10.0 < new_C < 200.0 and 2.0 < new_R < 60.0:
                        self.C = float(new_C)
                        self.R = float(new_R)
                        logger.info(f"[SysID] SUCCES (Gesplitste Methode): R={self.R:.2f}, C={self.C:.2f}")
                        # Sla model op en stop de training hier
                        # (de rest van K_emit etc. volgt na de if/else)
                    else:
                        logger.warning(
                            f"[SysID] Gesplitste methode gaf onrealistische waarden (R={new_R:.1f}, C={new_C:.1f}). Fallback.")
                        tau_inv = None  # Forceer fallback
                else:
                    logger.warning("[SysID] Negatieve coëfficiënt gevonden in verwarmingsfase. Fallback.")
                    tau_inv = None  # Forceer fallback
            else:
                logger.warning("[SysID] Niet genoeg data voor verwarmingsfase. Fallback.")
                tau_inv = None  # Forceer fallback

        # ==========================================================
        # STRATEGIE 2: Gecombineerde Meervoudige Regressie (Fallback)
        # ==========================================================
        if tau_inv is None:  # Als de vorige methode niet slaagde
            logger.info("[SysID] Poging 2: Fallback naar gecombineerde multivariate regressie.")
            df_model_data = df_1h.dropna(subset=['dT_next', 'wp_output', 'delta_T_env'])
            mask = (df_model_data["wp_output"] > 0.1) | (np.abs(df_model_data["delta_T_env"]) > 3.0)
            df_model_data = df_model_data[mask]

            if len(df_model_data) > 20:
                X = df_model_data[["wp_output", "delta_T_env"]]
                y = df_model_data["dT_next"]  # Let op: dT_next, niet dT_dt
                model = LinearRegression(fit_intercept=False).fit(X, y)

                beta_1 = model.coef_[0]  # Moet positief zijn (1/C)
                beta_2 = -model.coef_[1]  # Moet positief zijn (1/(R*C))

                if beta_1 > 0.001 and beta_2 > 0.001:
                    C_est = 1 / beta_1
                    R_est = beta_1 / beta_2
                    self.C = float(np.clip(C_est, 10.0, 200.0))
                    self.R = float(np.clip(R_est, 2.0, 60.0))
                    logger.info(f"[SysID] SUCCES (Fallback Methode): R={self.R:.2f}, C={self.C:.2f}")
                else:
                    logger.error(
                        f"[SysID] Beide trainingsmethodes mislukt. Coëfficiënten onlogisch. Defaults worden behouden.")
            else:
                logger.error(f"[SysID] Beide trainingsmethodes mislukt door te weinig data. Defaults worden behouden.")

        df_15m = df_proc.set_index("timestamp").resample("15min").mean().dropna().reset_index()

        # --- LEER DE TRAAGHEID (LAG) VAN DE VLOER ---
        df_lag = df_15m.copy()
        df_lag['dT_room'] = df_lag['room_temp'].diff()

        best_lag = 0
        best_corr = -1.0

        # Test vertragingen van 1 kwartier (15m) tot 16 kwartieren (4 uur)
        for lag in range(1, 17):
            df_lag[f'wp_lag_{lag}'] = df_lag['wp_output'].shift(lag)
            corr = df_lag['dT_room'].corr(df_lag[f'wp_lag_{lag}'])
            if pd.notna(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag

        self.ufh_lag_steps = int(np.clip(best_lag, 1, 12))  # Minimaal 30m, Max 3 uur

        # --- LEER K_emit & K_tank (Als fallback of validatie) ---
        mask_ufh = (df_15m["hvac_mode"] == HvacMode.HEATING.value) & (df_15m["wp_output"] > 0.7)
        df_ufh = df_15m[mask_ufh].copy()
        if len(df_ufh) > 20:
            t_avg_water = (df_ufh["supply_temp"] + df_ufh["return_temp"]) / 2
            delta_T_emit = t_avg_water - df_ufh["room_temp"]
            valid = delta_T_emit > 0.5
            if valid.any():
                self.K_emit = float(np.clip((df_ufh.loc[valid, "wp_output"] / delta_T_emit[valid]).median(), 0.05, 1.5))

        mask_dhw = (df_15m["hvac_mode"] == HvacMode.DHW.value) & (df_15m["wp_output"] > 1.5)
        df_dhw = df_15m[mask_dhw].copy()
        if len(df_dhw) > 10:
            t_tank = (df_dhw["dhw_top"] + df_dhw["dhw_bottom"]) / 2
            delta_T_hx = ((df_dhw["supply_temp"] + df_dhw["return_temp"]) / 2) - t_tank
            valid_idx = delta_T_hx > 2.0
            if valid_idx.any():
                self.K_tank = float(
                    np.clip((df_dhw.loc[valid_idx, "wp_output"] / delta_T_hx[valid_idx]).median(), 0.15, 2.0))

        joblib.dump(
            {"R": self.R, "C": self.C, "K_emit": self.K_emit, "K_tank": self.K_tank, "K_loss_dhw": self.K_loss_dhw,
             "ufh_lag_steps": self.ufh_lag_steps}, self.path)


# =========================================================
# 2a. HYDRAULIC PREDICTOR (NIEUW: Zelflerend debiet/aanvoer)
# =========================================================
class HydraulicPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.is_fitted = False
        self.model_supply_ufh = None
        self.model_supply_dhw = None
        self.model_delta_ufh = None
        # Features: Thermisch Vermogen, Buiten Temp, Doel Temp (Kamer of Boiler)
        self.features = ["wp_output", "temp", "sink_temp"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.model_supply_ufh = data["supply_ufh"]
                self.model_supply_dhw = data["supply_dhw"]
                self.model_delta_ufh = data["delta_ufh"]
                self.is_fitted = True
                logger.info("[Hydraulic] Zelflerend aanvoertemperatuur model geladen.")
            except Exception as e:
                logger.warning(f"[Hydraulic] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy()

        # Basis filter: Alleen rijen waar de warmtepomp echt draait en warmte levert
        # Delta T moet positief zijn (Aanvoer > Retour)
        df["delta_t"] = df["supply_temp"] - df["return_temp"]
        df = df[df["delta_t"] > 1.0].copy() # Filter ruis en stilstand

        # Voeg sink_temp toe voor de features
        conditions = [df["hvac_mode"] == HvacMode.HEATING.value, df["hvac_mode"] == HvacMode.DHW.value]
        choices_sink = [df["room_temp"], df["dhw_bottom"]]
        df["sink_temp"] = np.select(conditions, choices_sink, default=20.0)

        # Gebruik fysieke berekening voor training data labeling
        df["wp_output"] = np.select(conditions, [df["delta_t"] * FACTOR_UFH, df["delta_t"] * FACTOR_DHW], default=0.0)

        # --- Filteren voor UFH (Verwarming) ---
        mask_ufh = (
            (df["hvac_mode"] == HvacMode.HEATING.value) &
            (df["wp_output"] > 0.6) &
            (df["supply_temp"] > df["room_temp"] + 1.0) & # Moet warmer zijn dan de kamer
            (df["supply_temp"].between(22.0, 30.0))      # Realistische grenzen
        )
        df_ufh = df[mask_ufh].dropna(subset=self.features + ["supply_temp", "delta_t"])

        if len(df_ufh) > 15: # Iets meer data nodig voor betrouwbaarheid
            self.model_supply_ufh = RandomForestRegressor(n_estimators=50, max_depth=6).fit(df_ufh[self.features], df_ufh["supply_temp"])
            self.model_delta_ufh = RandomForestRegressor(n_estimators=50, max_depth=6).fit(df_ufh[self.features], df_ufh["delta_t"])
            logger.info(f"[Hydraulic] UFH getraind op {len(df_ufh)} schone datapunten.")

        # --- Filteren voor DHW (Boiler) ---
        mask_dhw = (
            (df["hvac_mode"] == HvacMode.DHW.value) &
            (df["wp_output"] > 0.9) &
            (df["supply_temp"] > df["dhw_bottom"] + 2.0) & # Moet warmer zijn dan de tank
            (df["supply_temp"].between(30.0, 65.0))       # Realistische grenzen
        )
        df_dhw = df[mask_dhw].dropna(subset=self.features + ["supply_temp"])

        if len(df_dhw) > 10:
            self.model_supply_dhw = RandomForestRegressor(n_estimators=50, max_depth=6).fit(df_dhw[self.features], df_dhw["supply_temp"])
            logger.info(f"[Hydraulic] DHW getraind op {len(df_dhw)} schone datapunten.")

        self.is_fitted = True
        joblib.dump({"supply_ufh": self.model_supply_ufh, "supply_dhw": self.model_supply_dhw, "delta_ufh": self.model_delta_ufh}, self.path)

    def predict_supply(self, mode, p_th, t_out, t_sink):
        """Voorspelt aanvoer. Nu met betere natuurkundige grenzen."""

        if mode == "UFH":
            # Realistischer voor een heel huis: K_emit rond 0.6 t/m 1.0
            # base_lift is hoeveel graden we boven de kamertemp moeten zitten
            base_lift = (p_th / 0.8) + (p_th / FACTOR_UFH / 2)
            physical_guess = t_sink + base_lift

            # Veiligheid: Vloer nooit boven 40 graden (tenzij ML anders bewijst)
            max_safe = 30.0
            min_lift = 2.0  # Altijd 2 graden boven kamer
        else:
            # DHW: Spiraal heeft minder oppervlak, dus meer lift nodig
            base_lift = (p_th / 0.4) + (p_th / FACTOR_DHW / 2)
            physical_guess = t_sink + base_lift
            max_safe = 60.0
            min_lift = 4.0  # Altijd 4 graden boven tank

        # Pas ML toe als het er is
        if self.is_fitted:
            data = pd.DataFrame([[p_th, t_out, t_sink]], columns=self.features)
            try:
                if mode == "UFH" and self.model_supply_ufh:
                    val = float(self.model_supply_ufh.predict(data)[0])
                    # Mix: We vertrouwen ML voor 70%, fysica voor 30%
                    prediction = 0.7 * val + 0.3 * physical_guess
                elif mode == "DHW" and self.model_supply_dhw:
                    val = float(self.model_supply_dhw.predict(data)[0])
                    prediction = 0.7 * val + 0.3 * physical_guess
                else:
                    prediction = physical_guess
            except:
                prediction = physical_guess
        else:
            prediction = physical_guess

        # HARD RAIL: De aanvoer moet altijd hoger zijn dan de sink + min_lift,
        # maar mag nooit boven de veiligheidsgrens uitkomen.
        return np.clip(prediction, t_sink + min_lift, max_safe)

    def predict_delta(self, mode, p_th, t_out, t_sink):
        if self.is_fitted and mode == "UFH" and self.model_delta_ufh:
            data = pd.DataFrame([[p_th, t_out, t_sink]], columns=self.features)
            return float(self.model_delta_ufh.predict(data)[0])
        # Fysische fallback: DeltaT = Vermogen / (FlowFactor)
        factor = FACTOR_UFH if mode == "UFH" else FACTOR_DHW
        return p_th / factor if p_th > 0 else 0.0
# =========================================================
# 3. ML RESIDUALS
# =========================================================
class MLResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.features = ["temp", "solar", "wind", "hour_sin", "hour_cos", "day_sin", "day_cos", "doy_sin", "doy_cos"]

    def train(self, df, R, C, is_dhw=False, K_loss_dhw=0.0):
        self.R = R
        self.C = C
        self.K_loss_dhw = K_loss_dhw
        df_proc = df.copy().sort_values("timestamp")

        # Bereken wp_output voor de hele set
        df_proc["delta_t_water"] = (
            df_proc["supply_temp"] - df_proc["return_temp"]
        ).clip(lower=0)
        df_proc["wp_output"] = np.where(
            df_proc["hvac_mode"] == HvacMode.DHW.value,
            df_proc["delta_t_water"] * FACTOR_DHW,
            np.where(
                df_proc["hvac_mode"] == HvacMode.HEATING.value,
                df_proc["delta_t_water"] * FACTOR_UFH,
                0.0,
            ),
        )

        dt_hours = df_proc["timestamp"].diff().dt.total_seconds().shift(-1) / 3600

        if not is_dhw:
            # UFH: Alleen trainen op Heating of Standby (om afkoeling te leren)
            mask = (df_proc["hvac_mode"] == HvacMode.HEATING.value) | (
                df_proc["wp_output"] < 0.1
            )
            df_feat = df_proc[mask].copy()
            dt = dt_hours[mask]

            t_curr = df_feat["room_temp"]
            t_next = df_feat["room_temp"].shift(-1)
            p_heat = df_feat["wp_output"]
            t_out = df_feat["temp"]

            # Fysische basis: T_next = T_curr + (P_heat - Verlies) * dt / C
            t_model_next = t_curr + ((p_heat - (t_curr - t_out) / self.R) * dt / self.C)
        else:
            # DHW: JUIST trainen op ALLES (ook als HP uit staat) om sluipverbruik te leren
            df_feat = df_proc.copy()
            dt = dt_hours

            tank_cap = 0.232  # 200L boiler constant
            t_curr = (df_feat["dhw_top"] + df_feat["dhw_bottom"]) / 2
            t_next = t_curr.shift(-1)
            p_heat = df_feat["wp_output"]
            # Boiler verlies model: K_loss_dhw * (T_tank - T_room)
            t_room = df_feat["room_temp"]

            # Fysische basis inclusief het vaste K_loss uit je SysID
            t_model_next = (
                t_curr
                + (p_heat * dt / tank_cap)
                - (self.K_loss_dhw * (t_curr - t_room) * dt)
            )

        # Residu berekenen (K/u)
        target = (t_next - t_model_next) / dt

        # Ruis wegpoetsen
        if is_dhw:
            # Alles tussen -0.8 en +0.8 wordt 0.0
            target = np.where(np.abs(target) < 0.2, 0, target)
            target = np.where(target > 0, 0, target)
        else:
            target = np.where(np.abs(target) < 0.15, 0, target)

        df_feat = add_cyclic_time_features(df_feat, "timestamp")
        df_feat["solar"] = df_feat["pv_actual"]
        df_feat["target"] = target

        train_set = df_feat[self.features + ["target"]].dropna()

        # DHW kan grotere afwijkingen hebben door taps, dus ruimere grenzen.
        if is_dhw:
            # Vangt sensorfouten op, 0.5 zorgt dat we niet leren van HP-opwarmfouten
            train_set = train_set[train_set["target"].between(-1.0, 0.5)]
        else:
            # Vangt extreme situaties op zoals open ramen of directe zon op de sensor
            train_set = train_set[train_set["target"].between(-1.2, 1.2)]

        print("Debug ML Residuals Train Set:")
        print(train_set[["temp", "solar", "target"]].head(30))

        if len(train_set) > 10:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=15
            ).fit(train_set[self.features], train_set["target"])

            joblib.dump(self.model, self.path)

    def predict(self, forecast_df):
        if self.model is None:
            return np.zeros(len(forecast_df))
        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        df["solar"] = df.get("power_corrected", 0.0)
        for col in self.features:
            if col not in df.columns: df[col] = 0.0
        return self.model.predict(df[self.features])


# =========================================================
# 4. THERMAL MPC (100% Lineair, Zonder Frequentie)
# =========================================================
class ThermalMPC:
    def __init__(self, ident, perf_map, hydraulic):
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic # NIEUW: Het hydraulische model
        self.horizon = 96
        self.dt = 0.25

        # Wordt alleen nog gebruikt voor logging/debug, logica zit nu in hydraulic
        self.FACTOR_UFH = (18.0 / 60.0) * 4.186
        self.FACTOR_DHW = (19.0 / 60.0) * 4.186

        self._build_problem()

    def _build_problem(self):
        T = self.horizon

        # --- PARAMETERS ---
        self.P_t_room_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()

        # Prijzen en Net
        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_export_prices = cp.Parameter(T, nonneg=True)
        self.P_solar = cp.Parameter(T, nonneg=True)
        self.P_base_load = cp.Parameter(T, nonneg=True)

        # Weer en Comfort
        self.P_temp_out = cp.Parameter(T)
        self.P_room_min = cp.Parameter(T, nonneg=True)
        self.P_room_max = cp.Parameter(T, nonneg=True)
        self.P_dhw_min = cp.Parameter(T, nonneg=True)
        self.P_dhw_max = cp.Parameter(T, nonneg=True)

        # Dynamisch berekende fysica (COP & Limits)
        self.P_cop_ufh = cp.Parameter(T, nonneg=True)
        self.P_cop_dhw = cp.Parameter(T, nonneg=True)
        self.P_max_pel = cp.Parameter(T, nonneg=True)  # Max elektrisch vermogen
        self.P_min_pel = cp.Parameter(T, nonneg=True)  # Min elektrisch vermogen (pendelgrens)

        # Traagheid: Warmte die al in de vloer zit uit het verleden
        # De grootte van deze parameter hangt af van self.ident.ufh_lag_steps
        self.P_hist_heat = cp.Parameter(self.ident.ufh_lag_steps, nonneg=True)

        # --- VARIABELEN (Puur Elektrisch kW) ---
        self.p_el_ufh = cp.Variable(T, nonneg=True)
        self.p_el_dhw = cp.Variable(T, nonneg=True)

        # Booleans voor status (AAN/UIT en exclusiviteit)
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)

        # Grid interactie
        self.p_grid = cp.Variable(T, nonneg=True)
        self.p_export = cp.Variable(T, nonneg=True)
        self.p_solar_self = cp.Variable(T, nonneg=True)

        # Temperaturen (States)
        self.t_room = cp.Variable(T + 1)
        self.t_dhw = cp.Variable(T + 1, nonneg=True)

        # Slacks voor comfort (zodat de solver niet crasht als het onmogelijk is)
        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_room_high = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)

        # --- CONSTRAINTS ---
        R, C = self.ident.R, self.ident.C
        lag = self.ident.ufh_lag_steps

        constraints = [
            self.t_room[0] == self.P_t_room_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        # --- TRAAGHEID LOGICA ---
        # We bouwen een vector op met thermisch vermogen:
        # [Historie (-lag), ..., Historie (-1), Toekomst (0), Toekomst (1), ...]
        p_th_ufh_future = cp.multiply(self.p_el_ufh, self.P_cop_ufh)
        p_th_ufh_lagged = cp.hstack([self.P_hist_heat, p_th_ufh_future])

        for t in range(T):
            p_el_wp = self.p_el_ufh[t] + self.p_el_dhw[t]
            p_th_dhw_now = self.p_el_dhw[t] * self.P_cop_dhw[t]

            # De warmte die NU de kamer inkomt, is 'lag' stappen geleden gemaakt
            active_room_heat = p_th_ufh_lagged[t]

            constraints += [
                # 1. Stroombalans
                p_el_wp + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],

                # 2. Thermische Transities
                # Kamer (Met vertraging!)
                self.t_room[t + 1] == self.t_room[t] + (
                        (active_room_heat - (self.t_room[t] - self.P_temp_out[t]) / R) * self.dt / C
                ),

                # DHW (Direct, met stilstandsverlies)
                self.t_dhw[t + 1] == self.t_dhw[t] + (
                        (p_th_dhw_now * self.dt) / 0.232  # 0.232 = Cap voor 200L boiler
                        - (self.t_dhw[t] - self.t_room[t]) * (self.ident.K_loss_dhw * self.dt)
                ),

                # 3. Fysieke Grenzen & Exclusiviteit
                self.ufh_on[t] + self.dhw_on[t] <= 1,

                # Maximaal vermogen (berekend in solve o.b.v. Supply Temp limiet)
                self.p_el_ufh[t] <= self.ufh_on[t] * self.P_max_pel[t],
                self.p_el_dhw[t] <= self.dhw_on[t] * self.P_max_pel[t],

                # Minimaal vermogen (om pendelen te voorkomen)
                self.p_el_ufh[t] >= self.ufh_on[t] * self.P_min_pel[t],
                self.p_el_dhw[t] >= self.dhw_on[t] * self.P_min_pel[t],

                # 4. Comfort Grenzen (Soft constraints)
                self.t_room[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_room[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] <= self.P_dhw_max[t], # Harde max voor veiligheid
            ]

        # --- OBJECTIVE FUNCTION ---
        # Kosten minimaliseren (Grid Import * Prijs - Export * Prijs)
        net_cost = cp.sum(cp.multiply(self.p_grid, self.P_prices) - cp.multiply(self.p_export, self.P_export_prices)) * self.dt

        # Comfort penalty (zwaar wegen zodat hij alleen afwijkt als het echt moet)
        comfort = cp.sum(self.s_room_low * 20.0 + self.s_dhw_low * 15.0 + self.s_room_high * 5.0)

        # Switching penalty (voorkom aan/uit knipperlicht gedrag)
        switching = cp.sum(cp.abs(self.ufh_on[1:] - self.ufh_on[:-1])) * 1.5 + \
                    cp.sum(cp.abs(self.dhw_on[1:] - self.dhw_on[:-1])) * 25.0

        self.problem = cp.Problem(cp.Minimize(net_cost + comfort + switching), constraints)

    def _get_targets(self, now, T):
        """Bepaal de comfort-grenzen per tijdslot"""
        r_min, r_max, d_min, d_max = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        for t in range(T):
            fut_time = now + timedelta(hours=t * self.dt)
            h = fut_time.hour

            # Vloerverwarming: Overdag en 's avonds comfort, 's nachts iets lager
            r_min[t] = 20.0 if 7 <= h <= 23 else 19.0
            r_max[t] = 21.5 if 8 <= h <= 20 else 20.5 # Sta lichte "overheating" toe bij zon

            # Boiler: Warm hebben voor de avonddouche (17:00-21:00)
            # Rest van de dag mag hij zakken tot 40, maar 's middags boosten we vaak op zon
            d_min[t] = 48.0 if 16 <= h <= 21 else 35.0
            d_max[t] = 55.0 # Max boiler temp (voor COP behoud)

        return r_min, r_max, d_min, d_max

    def solve(self, state, forecast_df, recent_history_df):
        T = self.horizon
        r_min, r_max, d_min, d_max = self._get_targets(state["now"], T)

        # Vul basis parameters
        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = (state["dhw_top"] + state["dhw_bottom"]) / 2.0
        self.P_temp_out.value = forecast_df.temp.values[:T]
        self.P_prices.value = np.full(T, 0.25)       # TODO: Dynamische prijzen hier koppelen
        self.P_export_prices.value = np.full(T, 0.07)
        self.P_solar.value = forecast_df.power_corrected.values[:T]
        self.P_base_load.value = forecast_df.load_corrected.values[:T]
        self.P_room_min.value, self.P_room_max.value = r_min, r_max
        self.P_dhw_min.value, self.P_dhw_max.value = d_min, d_max

        # --- FYSIEKE PARAMETERS VOORBEIDEIDING ---
        cop_u, cop_d, max_p, min_p = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        # We halen de geleerde parameters op
        R = self.ident.R             # Isolatie
        K_emit = self.ident.K_emit   # Vloerafgifte

        # Fysieke limiet beton (veiligheid, geen schatting maar harde grens materiaal)
        MAX_SUPPLY_LIMIT = 28.0

        for t in range(T):
            t_out = forecast_df.temp.values[t]
            t_room_target = r_min[t] # We willen minimaal dit halen

            # ---------------------------------------------------------------------
            # STAP 1: BEREKEN VEREISTE AANVOERTEMPERATUUR (INVERSE PHYSICS)
            # ---------------------------------------------------------------------
            # A. Hoeveel warmte verliest het huis bij deze buitentemp?
            #    Formula: P_loss = (T_binnen - T_buiten) / R
            heat_loss_kw = max(0, (t_room_target - t_out) / R)

            # VRAAG AAN ML: Als ik heat_loss_kw wil leveren, welke T_supply hoort daarbij?
            # We gebruiken de HydraulicPredictor in plaats van vaste flow formules.
            pred_supply = self.hydraulic.predict_supply("UFH", heat_loss_kw, t_out, t_room_target)

            # Veiligheid en Limieten
            calculated_supply_ufh = np.clip(pred_supply, t_room_target + 2.0, MAX_SUPPLY_LIMIT)

            # ---------------------------------------------------------------------
            # STAP 2: COP EN VERMOGEN BEPALEN
            # ---------------------------------------------------------------------
            cop_u[t] = self.perf_map.predict_cop(t_out, calculated_supply_ufh, HvacMode.HEATING.value)
            cop_d[t] = self.perf_map.predict_cop(t_out, d_max[t], HvacMode.DHW.value)

            min_kw_machine, max_kw_machine = self.perf_map.get_pel_limits(t_out)

            # Maximaal thermisch vermogen dat de vloer aankan bij MAX_SUPPLY_LIMIT
            # Hier gebruiken we nog wel K_emit als 'harde' veiligheidsgrens,
            # maar de target supply komt uit ML.
            delta_t_cap = max(0, MAX_SUPPLY_LIMIT - state["room_temp"])
            p_th_max_floor = delta_t_cap * K_emit

            # Vertaal naar elektrisch
            max_kw_floor = p_th_max_floor / cop_u[t] if cop_u[t] > 0 else 0

            # De limiet is de bottleneck (Machine of Vloer)
            max_p[t] = min(max_kw_machine, max_kw_floor)
            min_p[t] = min_kw_machine

            if max_p[t] < min_p[t]: max_p[t], min_p[t] = 0.0, 0.0

        self.P_cop_ufh.value = np.clip(cop_u, 2.0, 9.0)
        self.P_cop_dhw.value = np.clip(cop_d, 1.5, 5.0)
        self.P_max_pel.value = max_p
        self.P_min_pel.value = min_p

        # --- VUL HISTORISCHE WARMTE (LAG) ---
        lag = self.ident.ufh_lag_steps
        hist_heat = np.zeros(lag)

        # Check of dataframe geldig is en wp_output bevat
        if not recent_history_df.empty and 'wp_output' in recent_history_df.columns:
            vals = recent_history_df['wp_output'].tail(lag).values
            if len(vals) > 0:
                # Vul de array van achteren aan (meest recente historie achteraan)
                hist_heat[-len(vals):] = vals

        self.P_hist_heat.value = hist_heat

        # --- LOS HET PROBLEEM OP ---
        try:
            self.problem.solve(solver=cp.HIGHS)
        except Exception as e:
            logger.error(f"Solver exception: {e}")
            return None, None

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Solver status not optimal: {self.problem.status}")
            return None, None

        return self.p_el_ufh.value, self.p_el_dhw.value

    # def prepare_recent_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Zet ruwe sensor data van de afgelopen uren om naar 15-minuten blokken
    #     met de gemiddelde wp_output (thermisch vermogen in kW).
    #     """
    #     if raw_df is None or raw_df.empty:
    #         # Als er geen data is (bijv. bij de eerste keer opstarten), geef een leeg dataframe terug
    #         return pd.DataFrame(columns=['wp_output'])
    #
    #     df = raw_df.copy()
    #
    #     # Zorg dat de timestamp kolom een datetime object is
    #     if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    #         df['timestamp'] = pd.to_datetime(df['timestamp'])
    #
    #     # Bereken delta T (Aanvoer - Retour)
    #     df["delta_t"] = (df["supply_temp"] - df["return_temp"]).clip(lower=0.0)
    #
    #     # Bereken thermisch vermogen per meetpunt
    #     conditions = [
    #         df["hvac_mode"] == HVAC_MODE_HEATING,
    #         df["hvac_mode"] == HVAC_MODE_DHW,
    #     ]
    #     choices = [
    #         df["delta_t"] * FACTOR_UFH,
    #         df["delta_t"] * FACTOR_DHW,
    #     ]
    #     df["wp_output"] = np.select(conditions, choices, default=0.0)
    #
    #     # Zet de index op timestamp voor het resamplen
    #     df.set_index("timestamp", inplace=True)
    #
    #     # Resample naar gemiddelden per 15 minuten (15T of 15min)
    #     # Dit is cruciaal, want de MPC verwacht stappen van 15 minuten!
    #     df_15m = df.resample("15min").mean()
    #
    #     # Vul eventuele gaten op met 0 (als de WP uit stond en er geen data was)
    #     df_15m["wp_output"] = df_15m["wp_output"].fillna(0.0)
    #
    #     # Reset de index zodat we weer een normale DataFrame hebben (optioneel, maar netjes)
    #     df_15m.reset_index(inplace=True)
    #
    #     return df_15m[['timestamp', 'wp_output']]


# =========================================================
# 5. OPTIMIZER (Met fysieke Supply Temp bepaling)
# =========================================================
class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.hp_model_path)
        self.ident = SystemIdentificator(config.rc_model_path)

        # NIEUW: Hydraulisch model (Aanvoer/Retour/Flow gedrag leren)
        self.hydraulic = HydraulicPredictor(Path(config.hp_model_path).parent / "hydraulic_model.joblib")

        self.res_ufh = MLResidualPredictor(config.ufh_model_path)
        self.res_dhw = MLResidualPredictor(config.dhw_model_path)

        # Geef de hydraulic predictor mee aan MPC
        self.mpc = ThermalMPC(self.ident, self.perf_map, self.hydraulic)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)
        df = self.db.get_history(cutoff_date=cutoff)
        if df.empty: return

        self.perf_map.train(df)
        self.ident.train(df)
        self.hydraulic.train(df) # NIEUW: Train de hydrauliek
        self.mpc._build_problem()

    def resolve(self, context: Context):
        state = {
            "now": context.now,
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        cutoff = context.now - timedelta(hours=4)
        recent_history_df = self.db.get_history(cutoff_date=cutoff)

        temp_out = context.forecast_df.temp.values[0]

        # Krijg de ideale P_el (elektrische kW) voor de komende 24u
        p_el_ufh_plan, p_el_dhw_plan = self.mpc.solve(state, context.forecast_df, recent_history_df)

        if p_el_ufh_plan is None:
            return {"mode": "OFF", "target_pel_kw": 0.0, "target_supply_temp": 0.0}

        # Wat doen we NU (index 0)
        p_el_ufh_now = p_el_ufh_plan[0]
        p_el_dhw_now = p_el_dhw_plan[0]

        mode = "OFF"
        target_pel = 0.0
        target_supply_temp = 0.0

        # --- BEPALEN TARGETS MET ML (GEEN VASTE FORMULES MEER) ---
        if p_el_dhw_now > 0.1:
            mode = "DHW"
            target_pel = p_el_dhw_now

            # Bereken verwacht thermisch vermogen
            cop_now = self.perf_map.predict_cop(temp_out, context.dhw_bottom, HvacMode.DHW.value)
            p_th_dhw = target_pel * cop_now

            # Vraag aan HydraulicPredictor: Welke supply temp hoort bij dit vermogen?
            pred_supply = self.hydraulic.predict_supply("DHW", p_th_dhw, temp_out, context.dhw_bottom)
            target_supply_temp = pred_supply

        elif p_el_ufh_now > 0.1:
            mode = "UFH"
            target_pel = p_el_ufh_now

            # Bereken verwacht thermisch vermogen
            cop_now = self.perf_map.predict_cop(temp_out, context.room_temp, HvacMode.HEATING.value)
            p_th_ufh = target_pel * cop_now

            # Vraag aan HydraulicPredictor: Welke supply temp en delta T horen hierbij?
            pred_supply = self.hydraulic.predict_supply("UFH", p_th_ufh, temp_out, context.room_temp)
            pred_delta = self.hydraulic.predict_delta("UFH", p_th_ufh, temp_out, context.room_temp)

            logger.info(f"[Optimizer] UFH Active. P_th={p_th_ufh:.2f}kW -> Model Supply={pred_supply:.1f}, Delta={pred_delta:.1f}")

            target_supply_temp = pred_supply

        return {
            "mode": mode,
            "target_pel_kw": round(target_pel, 2),
            "target_supply_temp": round(target_supply_temp, 1),
            "status": self.mpc.problem.status
        }