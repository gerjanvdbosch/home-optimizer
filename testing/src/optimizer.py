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
        # Bereken output op basis van HVAC mode
        conditions = [
            df_proc["hvac_mode"] == HvacMode.HEATING.value,
            df_proc["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices = [df_proc["delta_t"] * FACTOR_UFH, df_proc["delta_t"] * FACTOR_DHW]
        df_proc["wp_output"] = np.select(conditions, choices, default=0.0)

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
        # Stap 1: Smooth de kamertemperatuur (voorkomt het "trapjes" effect van sensoren)
        # We nemen een rollend gemiddelde van 4 kwartieren (1 uur) om de trend te zien.
        df_15m['room_smooth'] = df_15m['room_temp'].rolling(window=4, center=True).mean()

        # Stap 2: Bereken temperatuurverschil over een uur (niet per kwartier, dat is te weinig)
        # Dit versterkt het signaal van opwarming.
        df_15m['dT_trend'] = df_15m['room_smooth'].diff(4)  # Verschil met 1 uur geleden

        best_lag = 0
        best_corr = -1.0

        # We testen vertragingen van 0 tot 4 uur
        for lag in range(0, 17):
            # We vergelijken de warmtepomp output van 'lag' kwartieren geleden
            # met de temperatuurtrend van NU.
            col_name = f'wp_lag_{lag}'
            df_15m[col_name] = df_15m['wp_output'].shift(lag)

            # Bereken correlatie
            corr = df_15m['dT_trend'].corr(df_15m[col_name])

            if pd.notna(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag

        # Als de correlatie te zwak is (< 0.1), is de data te rommelig.
        # Dan pakken we een veilige fallback van 8 kwartieren (2 uur).
        if best_corr < 0.1:
            logger.warning(
                f"[SysID] Geen duidelijke vloertraagheid gevonden (max corr={best_corr:.2f}). Fallback naar 2 uur.")
            self.ufh_lag_steps = 8
        else:
            self.ufh_lag_steps = int(np.clip(best_lag, 2, 16))  # Minimaal 30 min, max 4 uur
            logger.info(
                f"[SysID] Vloertraagheid gedetecteerd: {self.ufh_lag_steps * 15} minuten (Corr={best_corr:.2f})")

        # --- LEER K_emit & K_tank (Als fallback of validatie) ---
        mask_ufh = (
                (df_proc["hvac_mode"] == HvacMode.HEATING.value)
                & (df_proc["wp_output"] > 0.7)
                & (df_proc["supply_temp"] < 30)  # Niet te heet (voorkom overshoot data)
                & (df_proc["supply_temp"] > df_proc["room_temp"] + 1)  # Zinvolle delta T
        )
        df_ufh = df_proc[mask_ufh].copy()

        if len(df_ufh) > 20:
            print("Debug UFH Data Sample:")
            print(
                df_ufh[["timestamp", "room_temp", "supply_temp", "wp_output"]].head(30)
            )

            t_avg_water = (df_ufh["supply_temp"] + df_ufh["return_temp"]) / 2
            delta_T_emit = t_avg_water - df_ufh["room_temp"]
            valid = delta_T_emit > 0.5
            if valid.any():
                k_values = df_ufh.loc[valid, "wp_output"] / delta_T_emit[valid]
                # AANPASSING: Clip verhoogd naar 1.5 omdat je vloer meer vermogen aankan
                self.K_emit = float(np.clip(k_values.median(), 0.05, 1.5))

        # ============================
        # 4. Identificatie K_tank (Spiraalafgifte)
        # ============================
        mask_dhw = (
                (df_proc["hvac_mode"] == HvacMode.DHW.value)
                & (df_proc["wp_output"] > 1.0)
                & (df_proc["supply_temp"] > df_proc["dhw_bottom"] + 1)

        )
        df_dhw = df_proc[mask_dhw].copy()

        if len(df_dhw) > 10:
            print("Debug DHW Data Sample:")
            print(
                df_dhw[
                    [
                        "timestamp",
                        "dhw_top",
                        "dhw_bottom",
                        "supply_temp",
                        "return_temp",
                        "wp_output",
                    ]
                ].head(30)
            )
            t_avg_water = (df_dhw["supply_temp"] + df_dhw["return_temp"]) / 2
            delta_T_hx = t_avg_water - df_dhw["dhw_bottom"]

            # Alleen als water warmer is dan tank
            valid_idx = delta_T_hx > 2.0
            if valid_idx.any():
                k_values = df_dhw.loc[valid_idx, "wp_output"] / delta_T_hx[valid_idx]
                self.K_tank = float(np.clip(k_values.median(), 0.15, 2.0))

        # ============================
        # 5. Identificatie Stilstandsverlies DHW
        # ============================
        # Gebruik originele DF voor tijdstappen, want resample kan gaten maskeren
        df_loss = df_proc.sort_values("timestamp").copy()
        df_loss["t_tank"] = (df_loss["dhw_top"] + df_loss["dhw_bottom"]) / 2.0

        # FIX: Gebruik de kolom timestamp voor de diff, niet de index
        df_loss["dt_hours"] = df_loss["timestamp"].diff().dt.total_seconds() / 3600.0
        df_loss["dT_tank"] = df_loss["t_tank"].diff()

        mask_sb = (
                (df_loss["hvac_mode"] != HvacMode.DHW.value)
                & (df_loss["wp_output"] < 0.1)
                & (df_loss["dT_tank"] < 0)
                & (df_loss["dt_hours"] > 0.1)
                & (df_loss["dT_tank"] > -0.3) # douchen
                & (df_loss["dt_hours"] < 1.0)
        )

        df_sb = df_loss[mask_sb].copy()
        if len(df_sb) > 20:
            print("Debug Stilstandsverlies DHW Sample:")
            print(df_sb[["timestamp", "dT_tank", "dt_hours", "wp_output"]].head(30))

            # Fysica: dT/dt = -(1 / (R_tank * C_tank)) * (T_tank - T_room)
            # We weten C_tank (0.232 kWh/K voor 200L).
            # We berekenen de 'verliesfactor' per graad verschil met de kamer
            t_diff = (df_sb["t_tank"] - df_sb["room_temp"]).clip(lower=1.0)
            loss_factor_per_hour = -(df_sb["dT_tank"] / df_sb["dt_hours"]) / t_diff

            # K_loss_dhw wordt nu een dimensieloze factor (1/h) of 1/(R*C)
            self.K_loss_dhw = float(
                np.clip(loss_factor_per_hour.quantile(0.25), 0.001, 0.05)
            )
            # Betekenis: bij 0.01 verliest hij 1% van het temperatuurverschil met de kamer per uur.

        logger.info(
            f"[Optimizer] Final Trained SysID: R={self.R:.1f}, C={self.C:.1f}, "
            f"K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f}, K_loss={self.K_loss_dhw:.3f}"
        )
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

        # NIEUW: Geleerde fysieke parameters (met veilige defaults)
        self.learned_factor_ufh = FACTOR_UFH  # Flow (kW/K)
        self.learned_factor_dhw = FACTOR_DHW
        self.learned_lift_ufh = 0.5           # Minimaal verschil Retour -> Kamer
        self.learned_lift_dhw = 3.0           # Minimaal verschil Retour -> Tank

        self.features = ["wp_output", "temp", "sink_temp"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.model_supply_ufh = data.get("supply_ufh")
                self.model_supply_dhw = data.get("supply_dhw")

                # Laad geleerde fysieke parameters (indien aanwezig in bestand)
                self.learned_factor_ufh = data.get("factor_ufh", FACTOR_UFH)
                self.learned_factor_dhw = data.get("factor_dhw", FACTOR_DHW)
                self.learned_lift_ufh = data.get("lift_ufh", 0.5)
                self.learned_lift_dhw = data.get("lift_dhw", 3.0)

                self.is_fitted = True
                logger.info("[Hydraulic] Zelflerend aanvoertemperatuur model geladen.")
            except Exception as e:
                logger.warning(f"[Hydraulic] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy()
        df["delta_t"] = (df["supply_temp"] - df["return_temp"]).clip(lower=0.1) # Voorkom delen door nul

        # Bepaal sink temp
        conditions = [df["hvac_mode"] == HvacMode.HEATING.value, df["hvac_mode"] == HvacMode.DHW.value]
        choices_sink = [df["room_temp"], df["dhw_bottom"]]
        df["sink_temp"] = np.select(conditions, choices_sink, default=20.0)

        # Bereken vermogen op basis van vaste constanten (om te labelen)
        # of gebruik wp_output uit data als je een echte warmtemeter hebt.
        df["wp_output"] = np.select(conditions, [df["delta_t"] * FACTOR_UFH, df["delta_t"] * FACTOR_DHW], default=0.0)

        # --- Filteren voor UFH (Verwarming) ---
        mask_ufh = (
            (df["hvac_mode"] == HvacMode.HEATING.value) &
            (df["wp_output"] > 0.6) &
            (df["supply_temp"] > df["room_temp"] + 1.0) &
            (df["delta_t"] > 0.5)
        )
        df_ufh = df[mask_ufh].dropna(subset=self.features + ["supply_temp", "return_temp"])

        if len(df_ufh) > 15:
            # 1. Leer de FLOW Factor (Vermogen / DeltaT)
            # We pakken de mediaan omdat flow vaak constant is bij vloerverwarming
            self.learned_factor_ufh = (df_ufh["wp_output"] / df_ufh["delta_t"]).median()

            # 2. Leer de Minimale Lift (Retour - Kamer)
            # We pakken het 5e percentiel: wat is het minimale verschil dat we nodig hebben?
            # Dit vertegenwoordigt de situatie "net genoeg warmteafgifte".
            actual_lift = df_ufh["return_temp"] - df_ufh["room_temp"]
            self.learned_lift_ufh = max(0.2, actual_lift.quantile(0.05))

            # Train ML Model
            self.model_supply_ufh = RandomForestRegressor(n_estimators=50, max_depth=6).fit(df_ufh[self.features], df_ufh["supply_temp"])
            logger.info(f"[Hydraulic] UFH Geleerd: Factor={self.learned_factor_ufh:.2f} kW/K, MinLift={self.learned_lift_ufh:.2f}C")

        # --- DHW TRAINING & PARAMETER EXTRACTIE ---
        mask_dhw = (
            (df["hvac_mode"] == HvacMode.DHW.value) &
            (df["wp_output"] > 1.0) &
            (df["supply_temp"] > df["dhw_bottom"] + 2.0)
        )
        df_dhw = df[mask_dhw].dropna(subset=self.features + ["supply_temp", "return_temp"])

        if len(df_dhw) > 10:
            # 1. Leer Flow Factor
            self.learned_factor_dhw = (df_dhw["wp_output"] / df_dhw["delta_t"]).median()

            # 2. Leer Lift (Retour - TankBodem)
            actual_lift_dhw = df_dhw["return_temp"] - df_dhw["dhw_bottom"]
            self.learned_lift_dhw = max(1.0, actual_lift_dhw.quantile(0.05))

            # Train ML Model
            self.model_supply_dhw = RandomForestRegressor(n_estimators=50, max_depth=6).fit(df_dhw[self.features], df_dhw["supply_temp"])
            logger.info(f"[Hydraulic] DHW Geleerd: Factor={self.learned_factor_dhw:.2f} kW/K, MinLift={self.learned_lift_dhw:.2f}C")

        self.is_fitted = True
        # Sla alles op, inclusief de geleerde constanten
        joblib.dump({
            "supply_ufh": self.model_supply_ufh,
            "supply_dhw": self.model_supply_dhw,
            "factor_ufh": self.learned_factor_ufh,
            "factor_dhw": self.learned_factor_dhw,
            "lift_ufh": self.learned_lift_ufh,
            "lift_dhw": self.learned_lift_dhw
        }, self.path)

    def predict_supply(self, mode, p_th, t_out, t_sink):
        """Voorspelt aanvoer op basis van GELEERDE hydraulische eigenschappen."""

        if mode == "UFH":
            # Gebruik geleerde parameters (met fallback naar constants als training faalde)
            factor = self.learned_factor_ufh if self.is_fitted else FACTOR_UFH
            min_lift = self.learned_lift_ufh if self.is_fitted else 0.5

            max_safe = 30.0

            # Fysieke Delta T
            delta_t_calc = p_th / factor if p_th > 0 else 0.0

            # Fysieke gok: Kamer + Geleerde Lift + Geleerde DeltaT
            physical_guess = t_sink + min_lift + delta_t_calc + 1.0

            # Harde ondergrens
            min_supply_hard = t_sink + min_lift + delta_t_calc

        else: # DHW
            factor = self.learned_factor_dhw if self.is_fitted else FACTOR_DHW
            min_lift = self.learned_lift_dhw if self.is_fitted else 3.0
            max_safe = 60.0

            delta_t_calc = p_th / factor if p_th > 0 else 0.0
            physical_guess = t_sink + min_lift + delta_t_calc + 2.0
            min_supply_hard = t_sink + min_lift + delta_t_calc

        # ML Predictie (Mix)
        prediction = physical_guess
        if self.is_fitted:
            # We vragen het model wat het denkt, op basis van historische data
            data = pd.DataFrame([[p_th, t_out, t_sink]], columns=self.features)
            try:
                if mode == "UFH" and self.model_supply_ufh:
                    val = float(self.model_supply_ufh.predict(data)[0])
                    # Mix: 70% ML (leert leidingverliezen etc), 30% Fysica
                    prediction = 0.7 * val + 0.3 * physical_guess
                elif mode == "DHW" and self.model_supply_dhw:
                    val = float(self.model_supply_dhw.predict(data)[0])
                    prediction = 0.7 * val + 0.3 * physical_guess
            except:
                pass  # Fallback naar physical_guess bij error

        logger.debug(f"[Hydraulic] Predict Supply: Mode={mode} P_th={p_th:.2f} T_out={t_out:.1f} T_sink={t_sink:.1f} => Pred={prediction:.1f} (Phys={physical_guess:.1f}, MinHard={min_supply_hard:.1f}) Val={val:.1f}")
        # STAP 5: Final Check & Clip
        # De voorspelling mag nooit lager zijn dan de fysieke ondergrens (min_supply_hard)
        # En nooit hoger dan de veiligheidsgrens (max_safe)
        return np.clip(prediction, min_supply_hard, max_safe)

    def predict_delta(self, mode, p_th, t_out, t_sink):
        # Gebruik de geleerde factor voor de delta T voorspelling
        factor = self.learned_factor_ufh if mode == "UFH" else self.learned_factor_dhw
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
        # FIX: Veilig data ophalen. PV is vaak 'pv_estimate' of 'power_corrected'
        df["solar"] = df.get("power_corrected", df.get("pv_estimate", 0.0))
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

        # FIX: We gebruiken nu VASTE profielen. Als hij aan staat, verbruikt hij DIT.
        # Geen vrijheid meer voor de solver om te kiezen tussen min/max.
        self.P_fixed_pel_ufh = cp.Parameter(T, nonneg=True)
        self.P_fixed_pel_dhw = cp.Parameter(T, nonneg=True)

        self.P_hist_heat = cp.Parameter(self.ident.ufh_lag_steps, nonneg=True)
        self.P_solar_gain = cp.Parameter(T) # BUGFIX: Toegevoegd voor zonnewarmte!

        # --- VARIABELEN ---
        # Binary ON/OFF is nu de enige echte keuzevariabele
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)

        # p_el wordt een afgeleide variabele (dependent), geen vrije keuze meer
        self.p_el_ufh = cp.Variable(T, nonneg=True)
        self.p_el_dhw = cp.Variable(T, nonneg=True)

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
                # BUGFIX: Solar gain is nu meegenomen in de kamerbalans!
                self.t_room[t + 1] == self.t_room[t] + (
                        (active_room_heat - (self.t_room[t] - self.P_temp_out[t]) / R) * self.dt / C
                ) + (self.P_solar_gain[t] * self.dt),

                # DHW (Direct, met stilstandsverlies)
                self.t_dhw[t + 1] == self.t_dhw[t] + (
                        (p_th_dhw_now * self.dt) / 0.232  # 0.232 = Cap voor 200L boiler
                        - (self.t_dhw[t] - self.t_room[t]) * (self.ident.K_loss_dhw * self.dt)
                ),

                # 3. Fysieke Grenzen & Exclusiviteit
                self.ufh_on[t] + self.dhw_on[t] <= 1,

                # 4. FIX: GEEN MODULATIE.
                # Het vermogen is EXACT gelijk aan het voorspelde profiel als hij AAN staat.
                self.p_el_ufh[t] == self.ufh_on[t] * self.P_fixed_pel_ufh[t],
                self.p_el_dhw[t] == self.dhw_on[t] * self.P_fixed_pel_dhw[t],

                # 5. Comfort
                self.t_room[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_room[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] <= self.P_dhw_max[t],
            ]

        # --- OBJECTIVE FUNCTION ---
        net_cost = cp.sum(cp.multiply(self.p_grid, self.P_prices) - cp.multiply(self.p_export, self.P_export_prices)) * self.dt

        # Comfort penalties
        comfort = cp.sum(self.s_room_low * 20.0 + self.s_dhw_low * 5.0 + self.s_room_high * 2.0)

        # Penalty voor ELKE KEER dat de modus van 0 naar 1 gaat (starten)
        start_dhw_penalty = cp.sum(cp.pos(self.dhw_on[1:] - self.dhw_on[:-1])) * 10.0  # Zeer zware straf
        start_ufh_penalty = cp.sum(cp.pos(self.ufh_on[1:] - self.ufh_on[:-1])) * 0.5

        switching = start_ufh_penalty + start_dhw_penalty

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
            d_min[t] = 50.0 if 16 <= h <= 21 else 10.0
            d_max[t] = 55.0 # Max boiler temp (voor COP behoud)

        return r_min, r_max, d_min, d_max

    def solve(self, state, forecast_df, recent_history_df, solar_gains):
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
        self.P_solar_gain.value = solar_gains[:T]

        cop_u, cop_d = np.zeros(T), np.zeros(T)

        # Arrays voor het VASTE vermogen (geleerd gedrag)
        fixed_p_ufh, fixed_p_dhw = np.zeros(T), np.zeros(T)

        MAX_SUPPLY_LIMIT = 30.0

        for t in range(T):
            t_out = forecast_df.temp.values[t]
            t_room_target = r_min[t]

            # 1. Warmteverlies bepalen voor dit timeslot
            heat_loss_kw = max(0, (t_room_target - t_out) / self.ident.R)

            # 2. Hydraulische predictie (Supply Temp)
            pred_supply = self.hydraulic.predict_supply("UFH", heat_loss_kw, t_out, t_room_target)
            calculated_supply_ufh = np.clip(pred_supply, t_room_target + 2.0, MAX_SUPPLY_LIMIT)

            # 3. COP en Limits
            cop_u[t] = self.perf_map.predict_cop(t_out, calculated_supply_ufh, HvacMode.HEATING.value)
            cop_d[t] = self.perf_map.predict_cop(t_out, d_max[t], HvacMode.DHW.value)

            min_kw_machine, max_kw_machine = self.perf_map.get_pel_limits(t_out)

            # 4. FIX UFH VERMOGEN
            # Bereken wat puur fysisch nodig is om het verlies te dekken
            req_pel_loss = heat_loss_kw / cop_u[t] if cop_u[t] > 0 else 0

            # Fysieke limiet van de vloer
            delta_t_cap = max(0, MAX_SUPPLY_LIMIT - state["room_temp"])
            p_th_max_floor = delta_t_cap * self.ident.K_emit
            max_kw_floor = p_th_max_floor / cop_u[t] if cop_u[t] > 0 else 0

            # KEUZE LOGICA:
            # We nemen het MAXIMUM van:
            # A) Het geleerde 'Steady State' minimum (min_kw_machine)
            # B) Wat er nodig is om het verlies te dekken (req_pel_loss)
            # Dit voorkomt dat hij te zacht draait als het koud is, maar voorkomt ook
            # dat hij naar MAX schiet (2kW) als dat niet nodig is.
            target_p_ufh = max(min_kw_machine, req_pel_loss)

            # Clip dit target wel altijd binnen de machinegrenzen en vloergrenzen
            fixed_p_ufh[t] = np.clip(target_p_ufh, min_kw_machine, min(max_kw_machine, max_kw_floor))

            # DHW blijft op Max Power (Boilers doen dat meestal)
            fixed_p_dhw[t] = max_kw_machine

            # Fallback (als 0, dan kan hij niet aan)
            if fixed_p_ufh[t] < 0.1: fixed_p_ufh[t] = 0.0
            if fixed_p_dhw[t] < 0.1: fixed_p_dhw[t] = 0.0

        self.P_cop_ufh.value = np.clip(cop_u, 2.0, 9.0)
        self.P_cop_dhw.value = np.clip(cop_d, 1.5, 5.0)

        self.P_fixed_pel_ufh.value = fixed_p_ufh
        self.P_fixed_pel_dhw.value = fixed_p_dhw

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
        raw_hist = self.db.get_history(cutoff_date=cutoff)

        # BUGFIX: Calculate wp_output before passing to MPC so lag works properly!
        recent_history_df = pd.DataFrame()
        if not raw_hist.empty:
            raw_hist = raw_hist.copy() # BUGFIX: Voorkomt SetWithCopyWarnings
            raw_hist["delta_t"] = (raw_hist["supply_temp"] - raw_hist["return_temp"]).clip(lower=0.0)
            raw_hist["wp_output"] = np.where(
                raw_hist["hvac_mode"] == HvacMode.HEATING.value, raw_hist["delta_t"] * FACTOR_UFH,
                np.where(raw_hist["hvac_mode"] == HvacMode.DHW.value, raw_hist["delta_t"] * FACTOR_DHW, 0.0)
            )
            raw_hist.set_index("timestamp", inplace=True)
            # BUGFIX: numeric_only=True weghalen warning
            recent_history_df = raw_hist.resample("15min").mean(numeric_only=True).fillna(0).reset_index()

        # FIX: Voorspel de zon-opwarming vooraf via het getrainde ML model
        solar_gains = self.res_ufh.predict(context.forecast_df)

        temp_out = context.forecast_df.temp.values[0]

        # Geef de solar_gains ook mee aan de solver
        p_el_ufh_plan, p_el_dhw_plan = self.mpc.solve(state, context.forecast_df, recent_history_df, solar_gains)

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
            cop_now = self.perf_map.predict_cop(temp_out, context.dhw_bottom, HvacMode.DHW.value)
            p_th_dhw = target_pel * cop_now
            target_supply_temp = self.hydraulic.predict_supply("DHW", p_th_dhw, temp_out, context.dhw_bottom)

        elif p_el_ufh_now > 0.1:
            mode = "UFH"
            target_pel = p_el_ufh_now
            cop_now = self.perf_map.predict_cop(temp_out, context.room_temp, HvacMode.HEATING.value)
            p_th_ufh = target_pel * cop_now
            target_supply_temp = self.hydraulic.predict_supply("UFH", p_th_ufh, temp_out, context.room_temp)

            logger.info(f"[Optimizer] UFH Active. Fixed Plan Power={target_pel:.2f}kW, Predicted Supply={target_supply_temp:.1f}C")

        return {
            "mode": mode,
            "target_pel_kw": round(target_pel, 2),
            "target_supply_temp": round(target_supply_temp, 1),
            "status": self.mpc.problem.status
        }