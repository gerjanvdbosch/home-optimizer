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
                logger.info("[Optimizer] Performance map geladen.")
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
            df["delta_t"]
            * FACTOR_UFH,  # Hier gebruiken we nog de vaste factor voor historische data verwerking
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
            logger.warning(
                "[Optimizer] Te weinig valide data voor performance map training."
            )
            return

        df_max = df_clean.copy()
        if len(df_max) > 50:
            # We trainen een simpele lineaire regressor voor de maximale stroom-enveloppe
            # Features: Buitentemperatuur en Tank/Kamer temperatuur
            self.max_pel_model = LinearRegression().fit(
                df_max[["temp", "sink_temp"]], df_max["wp_actual"]
            )
            # Voor de ondergrens (modulatie-stop) doen we hetzelfde
            self.min_pel_model = LinearRegression().fit(
                df_max[["temp", "sink_temp"]], df_max["wp_actual"]
            )

        # ============================
        # 2. COP model (Random Forest)
        # ============================
        self.cop_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42
        )
        self.cop_model.fit(df_clean[self.features_cop], df_clean["cop"])

        # ============================
        # 3. Leer de grenzen van de warmtepomp (Elektrisch vermogen)
        # ============================
        df_f = df_clean.copy()
        df_f["t_rounded"] = df_f["temp"].round()

        # # Wat is het maximale en minimale elektrische vermogen per buitentemperatuur?
        # max_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.99).reset_index()
        # min_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.05).reset_index()
        #
        # if not max_stats.empty:
        #     self.max_pel_model = LinearRegression().fit(
        #         max_stats[["t_rounded"]], max_stats["wp_actual"]
        #     )
        #     self.min_pel_model = LinearRegression().fit(
        #         min_stats[["t_rounded"]], min_stats["wp_actual"]
        #     )

        self.is_fitted = True
        joblib.dump(
            {
                "cop_model": self.cop_model,
                "max_pel_model": self.max_pel_model,
                "min_pel_model": self.min_pel_model,
            },
            self.path,
        )
        logger.info("[Optimizer] Performance Map getraind.")

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
        return np.clip(min_p, 0.5, 1.5), np.clip(max_p, 1.5, 4.0)


    def get_max_pel(self, t_out, t_sink):
        """Geeft de geleerde maximale stroomopname bij deze condities."""
        if not self.is_fitted or self.max_pel_model is None:
            return 2.5  # Veilig default

        data = pd.DataFrame([[t_out, t_sink]], columns=["temp", "sink_temp"])
        return float(np.clip(self.max_pel_model.predict(data)[0], 0.5, 5.0))

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
        self.C_tank = 0.232  # kWh/K (200L water)
        self.T_max_dhw = 58.0  # Max aanvoertemperatuur van de machine (voor veiligheid in optimalisatie)
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
                self.C_tank = data.get("C_tank", 0.232)
                self.T_max_dhw = data.get("T_max_dhw", 58.0)
                self.ufh_lag_steps = data.get("ufh_lag_steps", 4)
                logger.info(
                    f"[SysID] Geladen: Lag={self.ufh_lag_steps * 15}m, K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f} R={self.R:.1f} C={self.C:.1f} K_loss_dhw={self.K_loss_dhw:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to load SystemID: {e}")

    def train(self, df: pd.DataFrame):
        df_proc = df.copy().sort_values("timestamp")

        # Zorg dat alle relevante kolommen numeriek zijn
        for col in [
            "supply_temp",
            "return_temp",
            "room_temp",
            "temp",
            "hvac_mode",
            "pv_actual",
        ]:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

        # Bereken thermisch vermogen
        df_proc["delta_t"] = (df_proc["supply_temp"] - df_proc["return_temp"]).clip(
            lower=0.0
        )
        # Bereken output op basis van HVAC mode
        conditions = [
            df_proc["hvac_mode"] == HvacMode.HEATING.value,
            df_proc["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices = [df_proc["delta_t"] * FACTOR_UFH, df_proc["delta_t"] * FACTOR_DHW]
        df_proc["wp_output"] = np.select(conditions, choices, default=0.0)

        # Resample naar 1 uur voor stabiele thermische modellering
        df_1h = (
            df_proc.set_index("timestamp")
            .resample("1h")
            .mean()
            .dropna(subset=["room_temp", "temp", "pv_actual", "wp_output"])
            .reset_index()
        )

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
                    f"[SysID] Afkoelfase geïdentificeerd met {len(df_cool)} datapunten. Geschatte Tau (RC) = {tau:.1f} uur."
                )

                # Sanity Check: Tau moet binnen realistische grenzen vallen
                if not (20 < tau < 1000):
                    logger.warning(
                        f"[SysID] Tau waarde ({tau:.1f}) is onrealistisch. Fallback naar gecombineerde methode."
                    )
                    tau_inv = None  # Reset om de fallback te triggeren
            else:
                logger.warning(
                    "[SysID] Negatieve coëfficiënt gevonden in afkoelfase, data is waarschijnlijk te ruizig."
                )

        else:
            logger.warning(
                f"[SysID] Te weinig data voor afkoelfase ({len(df_cool)} punten). Fallback naar gecombineerde methode."
            )

        # --- STAP 1B: Verwarming (Bepaal C, en dan R) ---
        if tau_inv is not None:
            mask_heat = (df_1h["wp_output"] > 0.5) & (df_1h["dT_next"] > 0.01)
            df_heat = df_1h[mask_heat].copy()

            if len(df_heat) > 10:
                y_heat_adjusted = df_heat["dT_next"] + (
                    tau_inv * df_heat["delta_T_env"]
                )
                X_heat = df_heat[["wp_output"]]
                lr_heat = LinearRegression(fit_intercept=False).fit(
                    X_heat, y_heat_adjusted
                )

                if lr_heat.coef_[0] > 0:
                    new_C = 1.0 / lr_heat.coef_[0]
                    new_R = (1.0 / tau_inv) / new_C

                    if 10.0 < new_C < 200.0 and 2.0 < new_R < 60.0:
                        self.C = float(new_C)
                        self.R = float(new_R)
                        logger.info(
                            f"[SysID] SUCCES (Gesplitste Methode): R={self.R:.2f}, C={self.C:.2f}"
                        )
                        # Sla model op en stop de training hier
                        # (de rest van K_emit etc. volgt na de if/else)
                    else:
                        logger.warning(
                            f"[SysID] Gesplitste methode gaf onrealistische waarden (R={new_R:.1f}, C={new_C:.1f}). Fallback."
                        )
                        tau_inv = None  # Forceer fallback
                else:
                    logger.warning(
                        "[SysID] Negatieve coëfficiënt gevonden in verwarmingsfase. Fallback."
                    )
                    tau_inv = None  # Forceer fallback
            else:
                logger.warning(
                    "[SysID] Niet genoeg data voor verwarmingsfase. Fallback."
                )
                tau_inv = None  # Forceer fallback

        # ==========================================================
        # STRATEGIE 2: Gecombineerde Meervoudige Regressie (Fallback)
        # ==========================================================
        if tau_inv is None:  # Als de vorige methode niet slaagde
            logger.info(
                "[SysID] Poging 2: Fallback naar gecombineerde multivariate regressie."
            )
            df_model_data = df_1h.dropna(subset=["dT_next", "wp_output", "delta_T_env"])
            mask = (df_model_data["wp_output"] > 0.1) | (
                np.abs(df_model_data["delta_T_env"]) > 3.0
            )
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
                    logger.info(
                        f"[SysID] SUCCES (Fallback Methode): R={self.R:.2f}, C={self.C:.2f}"
                    )
                else:
                    logger.error(
                        "[SysID] Beide trainingsmethodes mislukt. Coëfficiënten onlogisch. Defaults worden behouden."
                    )
            else:
                logger.error(
                    "[SysID] Beide trainingsmethodes mislukt door te weinig data. Defaults worden behouden."
                )

        df_15m = (
            df_proc.set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna()
            .reset_index()
        )

        # --- LEER DE TRAAGHEID (LAG) VAN DE VLOER ---
        # Stap 1: Smooth de kamertemperatuur (voorkomt het "trapjes" effect van sensoren)
        # We nemen een rollend gemiddelde van 4 kwartieren (1 uur) om de trend te zien.
        df_15m["room_smooth"] = (
            df_15m["room_temp"].rolling(window=4, center=True).mean()
        )

        # Stap 2: Bereken temperatuurverschil over een uur (niet per kwartier, dat is te weinig)
        # Dit versterkt het signaal van opwarming.
        df_15m["dT_trend"] = df_15m["room_smooth"].diff(4)  # Verschil met 1 uur geleden

        best_lag = 0
        best_corr = -1.0

        # We testen vertragingen van 0 tot 4 uur
        for lag in range(0, 17):
            # We vergelijken de warmtepomp output van 'lag' kwartieren geleden
            # met de temperatuurtrend van NU.
            col_name = f"wp_lag_{lag}"
            df_15m[col_name] = df_15m["wp_output"].shift(lag)

            # Bereken correlatie
            corr = df_15m["dT_trend"].corr(df_15m[col_name])

            if pd.notna(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag

        # Als de correlatie te zwak is (< 0.1), is de data te rommelig.
        # Dan pakken we een veilige fallback van 8 kwartieren (2 uur).
        if best_corr < 0.1:
            logger.warning(
                f"[SysID] Geen duidelijke vloertraagheid gevonden (max corr={best_corr:.2f}). Fallback naar 2 uur."
            )
            self.ufh_lag_steps = 8
        else:
            self.ufh_lag_steps = int(
                np.clip(best_lag, 2, 16)
            )  # Minimaal 30 min, max 4 uur
            logger.info(
                f"[SysID] Vloertraagheid gedetecteerd: {self.ufh_lag_steps * 15} minuten (Corr={best_corr:.2f})"
            )

        # --- LEER K_emit & K_tank (Als fallback of validatie) ---
        mask_ufh = (
            (df_proc["hvac_mode"] == HvacMode.HEATING.value)
            & (df_proc["wp_output"] > 0.5)
            & (df_proc["supply_temp"] < 30)  # Niet te heet (voorkom overshoot data)
            & (df_proc["supply_temp"] > df_proc["room_temp"] + 1)  # Zinvolle delta T
        )
        df_ufh = df_proc[mask_ufh].copy()

        if len(df_ufh) > 20:
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
            & (df_proc["wp_output"] > 0.8)
            & (df_proc["supply_temp"] > df_proc["dhw_bottom"] + 1)
        )
        df_dhw = df_proc[mask_dhw].copy()

        if len(df_dhw) > 10:
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
            & (df_loss["dT_tank"] > -0.3)  # douchen
            & (df_loss["dt_hours"] < 1.0)
        )

        df_sb = df_loss[mask_sb].copy()
        if len(df_sb) > 20:
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

        # We kijken naar DHW runs en berekenen: C = Energie_in / Temperatuurstijging
        df_dhw = df_proc[df_proc["hvac_mode"] == HvacMode.DHW.value].copy()
        if len(df_dhw) > 10:
            # Wat is de hoogste supply_temp die we ooit in DHW mode hebben gezien?
            self.T_max_dhw = float(df_dhw["supply_temp"].quantile(0.999))
            logger.info(f"[SysID] Geleerde Max DHW Temp: {self.T_max_dhw:.1f}C")

        df_dhw_all = df_proc.copy()
        # Markeer elke keer dat de modus verandert
        df_dhw_all["run_id"] = (
            df_dhw_all["hvac_mode"] != df_dhw_all["hvac_mode"].shift()
        ).cumsum()

        # Filter alleen op de DHW regels
        df_dhw_runs = df_dhw_all[df_dhw_all["hvac_mode"] == HvacMode.DHW.value].copy()

        calculated_cs = []

        # Loop per run (bijv. elke dag 1 run)
        for run_id, run_data in df_dhw_runs.groupby("run_id"):
            if len(run_data) < 4:
                continue  # Te korte run (minder dan een uur)

            # FILTER 1: Tappen tijdens opwarmen
            # Als de bovenste sensor meer dan 1 graad daalt tijdens de run,
            # wordt er warm water verbruikt. Dit verpest de energiebalans.
            t_top_start = run_data["dhw_top"].iloc[0]
            if run_data["dhw_top"].min() < t_top_start - 1.0:
                continue  # Sla deze run over, data is vervuild door tapwatergebruik

            # Energie die erin ging (Vermogen * tijd in uren)
            dt_hours = (
                run_data["timestamp"].diff().dt.total_seconds().fillna(900) / 3600.0
            )

            total_energy_in = (run_data["wp_output"] * dt_hours).sum()

            # Temperatuurstijging (Einde - Begin)
            t_start = (
                run_data["dhw_top"].iloc[:2].mean()
                + run_data["dhw_bottom"].iloc[:2].mean()
            ) / 2.0

            # Sensoren lopen altijd wat achter op het water (traagheid van de huls).
            # We pakken het MAXIMUM van de laatste paar metingen in de run.
            t_end_top = run_data["dhw_top"].iloc[-3:].max()
            t_end_bot = run_data["dhw_bottom"].iloc[-3:].max()
            t_end = (t_end_top + t_end_bot) / 2.0

            delta_T_total = t_end - t_start

            # Alleen valideren als er serieus gestookt is (> 5 graden erbij en > 1 kWh erin)
            if delta_T_total > 5.0 and total_energy_in > 1.0:
                c_calc = total_energy_in / delta_T_total

                # Uitschieters filteren: Een huishoudelijke boiler is tussen de 100L en 300L.
                # 100L = ~0.116 kWh/K  |  300L = ~0.348 kWh/K
                if 0.100 < c_calc < 0.350:
                    calculated_cs.append(c_calc)
                    logger.info(
                        f"[SysID] DHW Run: C={c_calc:.3f} kWh/K, T_start={t_start:.1f}C, T_end={t_end:.1f}C, T_diff={delta_T_total:.1f}C, total_energy_in={total_energy_in:.1f}kWh"
                    )

        if len(calculated_cs) > 0:
            # Pak de mediaan om uitschieters te negeren
            c_median = float(np.median(calculated_cs))

            # Fysieke clip: we staan maximaal 300L toe (0.350 kWh/K)
            self.C_tank = float(np.clip(c_median, 0.150, 0.350))
            logger.info(
                f"[SysID] Geleerde Boiler Massa (Integraal): {self.C_tank:.3f} kWh/K (o.b.v. {len(calculated_cs)} runs)"
            )

        logger.info(
            f"[Optimizer] Final Trained SysID: R={self.R:.1f}, C={self.C:.1f}, "
            f"K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f}, K_loss={self.K_loss_dhw:.3f} C_tank={self.C_tank:.3f} T_max_dhw={self.T_max_dhw:.1f}"
        )
        joblib.dump(
            {
                "R": self.R,
                "C": self.C,
                "K_emit": self.K_emit,
                "K_tank": self.K_tank,
                "K_loss_dhw": self.K_loss_dhw,
                "C_tank": self.C_tank,
                "T_max_dhw": self.T_max_dhw,
                "ufh_lag_steps": self.ufh_lag_steps,
            },
            self.path,
        )


# =========================================================
# 2a. HYDRAULIC PREDICTOR (NIEUW: Zelflerend debiet/aanvoer)
# =========================================================
class HydraulicPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.is_fitted = False

        self.model_supply_ufh = None
        self.model_supply_dhw = None

        # Features: Elektrisch Vermogen, Buiten Temp, Sink Temp
        self.features = ["wp_actual", "temp", "sink_temp"]

        # NIEUW: Geleerde fysieke parameters (met veilige defaults)
        self.learned_factor_ufh = FACTOR_UFH  # Flow (kW/K)
        self.learned_factor_dhw = FACTOR_DHW
        self.learned_lift_ufh = 0.5  # Minimaal verschil Retour -> Kamer
        self.learned_lift_dhw = 3.0  # Minimaal verschil Retour -> Tank
        self.learned_ufh_slope = 0.4  # Veilige default
        self.learned_lift_dhw = 3.0
        self.dhw_delta_slope = 0.0
        self.dhw_delta_base = 5.0

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
                self.learned_ufh_slope = data.get("ufh_slope", 0.4)
                self.dhw_delta_slope = data.get("dhw_slope", 0.0)
                self.dhw_delta_base = data.get("dhw_base", 5.0)

                self.is_fitted = True
                logger.info(
                    f"[Hydraulic] Model geladen. Geleerde parameters: UFH Factor={self.learned_factor_ufh:.2f}, DHW Factor={self.learned_factor_dhw:.2f}, UFH Lift={self.learned_lift_ufh:.2f}C, UFH Slope={self.learned_ufh_slope:.2f}, DHW Lift={self.learned_lift_dhw:.2f}C, DHW Slope={self.dhw_delta_slope:.3f} Base={self.dhw_delta_base:.1f}"
                )
            except Exception as e:
                logger.warning(f"[Hydraulic] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy()

        # 1. Bereken fysieke Delta T en check sensoren
        df["delta_t"] = df["supply_temp"] - df["return_temp"]
        df = df[(df["delta_t"] > 0.1) & (df["delta_t"] < 15.0)].copy()

        # 2. Bepaal sink temp & HVAC mode logica
        conditions = [
            df["hvac_mode"] == HvacMode.HEATING.value,
            df["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices_sink = [df["room_temp"], df["dhw_bottom"]]
        df["sink_temp"] = np.select(conditions, choices_sink, default=20.0)

        # 3. Bereken Vermogen (p_th_raw) voor filtering
        # We gebruiken de vaste factors als referentie
        choices_p = [df["delta_t"] * FACTOR_UFH, df["delta_t"] * FACTOR_DHW]
        df["p_th_raw"] = np.select(conditions, choices_p, default=0.0)
        df["wp_output"] = df["p_th_raw"]  # Voor backwards compatibility features

        # =========================================================================
        # UFH TRAINING (VLOERVERWARMING) - Ongewijzigd, dit werkte goed
        # =========================================================================
        mask_ufh = (
            (df["hvac_mode"] == HvacMode.HEATING.value)
            & (df["p_th_raw"] > 0.5)
            & (df["supply_temp"] > df["room_temp"] + 1.0)
        )
        df_ufh = df[mask_ufh].dropna(
            subset=self.features + ["supply_temp", "return_temp"]
        )

        if len(df_ufh) > 15:
            self.learned_factor_ufh = (df_ufh["p_th_raw"] / df_ufh["delta_t"]).median()
            actual_lift = df_ufh["return_temp"] - df_ufh["room_temp"]
            self.learned_lift_ufh = max(0.2, actual_lift.quantile(0.05))

            mask_curve = (
                mask_ufh
                & (df["temp"] < 15.0)
                & (df["supply_temp"] > df["room_temp"] + 2.0)
            )
            df_curve = df[mask_curve].copy()
            if len(df_curve) > 50:
                delta_t_buiten = 20.0 - df_curve["temp"]
                extra_supply = (
                    df_curve["supply_temp"]
                    - df_curve["room_temp"]
                    - self.learned_lift_ufh
                )
                slopes = extra_supply / delta_t_buiten
                self.learned_ufh_slope = float(np.clip(slopes.median(), 0.1, 1.5))

            self.model_supply_ufh = RandomForestRegressor(
                n_estimators=50, max_depth=6
            ).fit(df_ufh[self.features], df_ufh["supply_temp"])
            logger.info(
                f"[Hydraulic] UFH Geleerd: Factor={self.learned_factor_ufh:.2f}, Slope={self.learned_ufh_slope:.2f}"
            )

            # =========================================================================
            # DHW TRAINING (ROBUUST & SIMPEL)
            # =========================================================================
            mask_dhw = (
                (df["hvac_mode"] == HvacMode.DHW.value)
                & (df["p_th_raw"] > 1.5)  # Alleen actief draaiend
                & (df["delta_t"] > 2.0)
                & (df["supply_temp"] > df["dhw_bottom"] + 3.0)
            )
            df_dhw = (
                df[mask_dhw]
                .dropna(subset=self.features + ["supply_temp", "return_temp"])
                .copy()
            )

            if len(df_dhw) > 10:
                actual_lift_dhw = df_dhw["return_temp"] - df_dhw["dhw_bottom"]

                self.learned_factor_dhw = (
                    df_dhw["wp_output"] / df_dhw["delta_t"]
                ).median()
                self.learned_lift_dhw = max(1.0, actual_lift_dhw.quantile(0.10))

                # Bepaal gemiddelden (robuust tegen uitschieters)
                median_delta_t = df_dhw["delta_t"].median()

                df_dhw["t_bin"] = df_dhw["temp"].round()
                top_powers = []
                top_temps = []

                for t_val, group in df_dhw.groupby("t_bin"):
                    if len(group) >= 3:
                        top_powers.append(group["p_th_raw"].quantile(0.90))
                        top_temps.append(t_val)

                df_tops = pd.DataFrame({"temp": top_temps, "p_th_raw": top_powers})

                power_slope = 0.0
                power_base = median_delta_t * self.learned_factor_dhw  # Fallback base

                if len(df_tops) >= 3:
                    X_power = df_tops[["temp"]]
                    y_power = df_tops["p_th_raw"]

                    reg = LinearRegression().fit(X_power, y_power)
                    power_slope = float(reg.coef_[0])
                    power_base = float(reg.intercept_)
                    logger.info(
                        f"[Hydraulic] DHW curve Linear: Slope={power_slope:.2f}, Base={power_base:.1f}"
                    )

                    # ==========================================================
                    # D. Validatie & Heuristische grenzen (Empirisch capaciteitsmodel)
                    # ==========================================================
                    # We accepteren de geleerde slope als het werkelijke systeemgedrag
                    # (inclusief interne modulatie en compressor-begrenzing door de warmtepomp-software).
                    # Dit is een identificatie van de operationele envelop, geen natuurwet.

                    # Heuristische modelkeuzes: begrens de helling om extreme extrapolatie
                    # buiten het gemeten temperatuurbereik te voorkomen.
                    power_slope = float(np.clip(power_slope, -0.15, 0.25))

                # Koppel terug aan de energiebalans via de effectieve, constante flow-factor
                self.dhw_delta_slope = power_slope / self.learned_factor_dhw
                self.dhw_delta_base = power_base / self.learned_factor_dhw

                # Veiligheidsheuristiek: garandeer een realistische minimum ΔT (en dus >0 P_th)
                # bij hoge buitentemperaturen (35 °C). Dit voorkomt dat de wiskundige solver
                # een nul-vermogen staat plant in de zomer.
                min_delta_at_35C = self.dhw_delta_base + (self.dhw_delta_slope * 35.0)
                if min_delta_at_35C < 1.5:
                    logger.warning(
                        "[Hydraulic] Geëxtrapoleerde curve duikt te laag bij 35C. Base heuristisch verschoven."
                    )
                    self.dhw_delta_base += 1.5 - min_delta_at_35C

                logger.info(
                    f"[Hydraulic] Empirisch DHW capaciteitsmodel geaccepteerd: PowerSlope={power_slope:.2f} -> DeltaSlope={self.dhw_delta_slope:.3f}"
                )

                # Train ML Model voor Supply Temp (dit model leert ook evt niet-lineair gedrag)
                self.model_supply_dhw = RandomForestRegressor(
                    n_estimators=50, max_depth=6
                ).fit(df_dhw[self.features], df_dhw["supply_temp"])

                logger.info(
                    f"[Hydraulic] DHW Final: Lift={self.learned_lift_dhw:.1f}C, Base DeltaT={self.dhw_delta_base:.1f}C"
                )

        self.is_fitted = True
        joblib.dump(
            {
                "supply_ufh": self.model_supply_ufh,
                "supply_dhw": self.model_supply_dhw,
                "factor_ufh": self.learned_factor_ufh,
                "factor_dhw": self.learned_factor_dhw,
                "lift_ufh": self.learned_lift_ufh,
                "lift_dhw": self.learned_lift_dhw,
                "ufh_slope": self.learned_ufh_slope,
                "dhw_slope": self.dhw_delta_slope,
                "dhw_base": self.dhw_delta_base,
            },
            self.path,
        )

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

        else:  # DHW
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
            except Exception:
                pass  # Fallback naar physical_guess bij error

        logger.debug(
            f"[Hydraulic] Predict Supply: Mode={mode} P_th={p_th:.2f} T_out={t_out:.1f} T_sink={t_sink:.1f} => Pred={prediction:.1f} (Phys={physical_guess:.1f}, MinHard={min_supply_hard:.1f}) Val={val:.1f}"
        )
        # STAP 5: Final Check & Clip
        # De voorspelling mag nooit lager zijn dan de fysieke ondergrens (min_supply_hard)
        # En nooit hoger dan de veiligheidsgrens (max_safe)
        return np.clip(prediction, min_supply_hard, max_safe)

    def predict_delta(self, mode, p_th):
        # Gebruik de geleerde factor voor de delta T voorspelling
        factor = self.learned_factor_ufh if mode == "UFH" else self.learned_factor_dhw
        return p_th / factor if p_th > 0 else 0.0

    def get_ufh_slope(self, t_out):
        """Geeft de extra graden aanvoer terug o.b.v. buitentemperatuur."""
        diff = max(0.0, 20.0 - t_out)
        return diff * self.learned_ufh_slope


# =========================================================
# 3. ML RESIDUALS
# =========================================================
class MLResidualPredictor:
    def __init__(self, path, R, C, K_loss_dhw, C_tank):
        self.path = Path(path)
        self.R = R
        self.C = C
        self.K_loss_dhw = K_loss_dhw
        self.C_tank = C_tank
        self.model = None
        self.features = [
            "temp",
            "solar",
            "wind",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]

    def train(self, df, is_dhw=False):
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

            t_curr = (df_feat["dhw_top"] + df_feat["dhw_bottom"]) / 2
            t_next = t_curr.shift(-1)
            p_heat = df_feat["wp_output"]
            # Boiler verlies model: K_loss_dhw * (T_tank - T_room)
            t_room = df_feat["room_temp"]

            # Fysische basis inclusief het vaste K_loss uit je SysID
            t_model_next = (
                t_curr
                + (p_heat * dt / self.C_tank)
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
            if col not in df.columns:
                df[col] = 0.0
        return self.model.predict(df[self.features])


# =========================================================
# 4. THERMAL MPC
# =========================================================
class ThermalMPC:
    def __init__(self, ident, perf_map, hydraulic):
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.horizon = 96
        self.dt = 0.25
        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        R, C = self.ident.R, self.ident.C

        self.P_t_room_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()
        self.P_init_ufh = cp.Parameter(nonneg=True)
        self.P_init_dhw = cp.Parameter(nonneg=True)
        self.P_comp_on_init = cp.Parameter(nonneg=True)  # Voor compressor starts

        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_export_prices = cp.Parameter(T, nonneg=True)
        self.P_solar = cp.Parameter(T, nonneg=True)
        self.P_base_load = cp.Parameter(T, nonneg=True)
        self.P_temp_out = cp.Parameter(T)
        self.P_room_min = cp.Parameter(T, nonneg=True)
        self.P_room_max = cp.Parameter(T, nonneg=True)
        self.P_dhw_min = cp.Parameter(T, nonneg=True)
        self.P_dhw_max = cp.Parameter(T, nonneg=True)
        self.P_strictness = cp.Parameter(T, nonneg=True)
        self.P_solar_gain = cp.Parameter(T)
        self.P_hist_heat = cp.Parameter(self.ident.ufh_lag_steps, nonneg=True)

        # Dynamic DPP Penalty Parameters
        self.P_cost_room_under = cp.Parameter(nonneg=True)
        self.P_cost_room_over = cp.Parameter(nonneg=True)
        self.P_cost_dhw_under = cp.Parameter(nonneg=True)
        self.P_val_terminal_room = cp.Parameter(nonneg=True)
        self.P_val_terminal_dhw = cp.Parameter(nonneg=True)

        # Power Curves (Slopes/Constanten)
        self.P_dhw_pel_slope = cp.Parameter(T);
        self.P_dhw_pel_const = cp.Parameter(T)
        self.P_dhw_pth_slope = cp.Parameter(T);
        self.P_dhw_pth_const = cp.Parameter(T)
        self.P_ufh_pel_slope = cp.Parameter(T);
        self.P_ufh_pel_const = cp.Parameter(T)
        self.P_ufh_pth_slope = cp.Parameter(T);
        self.P_ufh_pth_const = cp.Parameter(T)

        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.z_dhw = cp.Variable(T, nonneg=True)  # Big-M
        self.z_ufh = cp.Variable(T, nonneg=True)  # Big-M

        self.comp_start = cp.Variable(T, nonneg=True) # Voor afstraffen daadwerkelijke starts

        self.p_el_ufh = cp.Variable(T, nonneg=True)
        self.p_el_dhw = cp.Variable(T, nonneg=True)
        self.p_grid = cp.Variable(T, nonneg=True)
        self.p_export = cp.Variable(T, nonneg=True)
        self.p_solar_self = cp.Variable(T, nonneg=True)
        self.t_room = cp.Variable(T + 1)
        self.t_dhw = cp.Variable(T + 1, nonneg=True)
        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_room_high = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)

        constraints = [self.t_room[0] == self.P_t_room_init, self.t_dhw[0] == self.P_t_dhw_init]

        # Big-M Koppelingen (Vectorized)
        M = 65.0
        constraints += [
            self.z_dhw <= M * self.dhw_on, self.z_dhw <= self.t_dhw[:-1],
            self.z_dhw >= self.t_dhw[:-1] - M * (1 - self.dhw_on),
            self.z_ufh <= M * self.ufh_on, self.z_ufh <= self.t_room[:-1],
            self.z_ufh >= self.t_room[:-1] - M * (1 - self.ufh_on)
        ]

        # Compressor logica (Expressie)
        comp_on = self.ufh_on + self.dhw_on
        constraints +=[self.comp_start[0] >= comp_on[0] - self.P_comp_on_init]
        for t in range(1, T):
            constraints += [self.comp_start[t] >= comp_on[t] - comp_on[t - 1]]

        # Power Curves koppelen aan variabelen
        p_el_dh_expr = cp.multiply(self.P_dhw_pel_slope, self.z_dhw) + cp.multiply(self.P_dhw_pel_const, self.dhw_on)
        p_th_dh_expr = cp.multiply(self.P_dhw_pth_slope, self.z_dhw) + cp.multiply(self.P_dhw_pth_const, self.dhw_on)
        p_el_uf_expr = cp.multiply(self.P_ufh_pel_slope, self.z_ufh) + cp.multiply(self.P_ufh_pel_const, self.ufh_on)
        p_th_uf_expr = cp.multiply(self.P_ufh_pth_slope, self.z_ufh) + cp.multiply(self.P_ufh_pth_const, self.ufh_on)

        constraints += [self.p_el_ufh == p_el_uf_expr, self.p_el_dhw == p_el_dh_expr]

        # Traagheid UFH
        p_th_ufh_lagged = cp.hstack([self.P_hist_heat, p_th_uf_expr])

        for t in range(T):
            p_el_total = self.p_el_ufh[t] + self.p_el_dhw[t]

            constraints +=[
                # Stroombalans & Net/Solar grenzen
                p_el_total + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],

                # Thermische Balans
                self.t_room[t + 1] == self.t_room[t] + (
                    ((p_th_ufh_lagged[t] - (self.t_room[t] - self.P_temp_out[t]) / R) * self.dt / C)) + (self.P_solar_gain[t] * self.dt),
                self.t_dhw[t + 1] == self.t_dhw[t] + ((p_th_dh_expr[t] * self.dt) / self.ident.C_tank) - (
                    self.t_dhw[t] - self.t_room[t]) * (self.ident.K_loss_dhw * self.dt),

                # Exclusiviteit
                self.ufh_on[t] + self.dhw_on[t] <= 1,

                # Comfort limieten
                self.t_room[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_room[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] <= self.P_dhw_max[t],
            ]

        # --- Objective Function ---
        solar_bonus_rate = 0.03 # Extra "bonus" per kWh eigen zonne-energie om de machine naar zonne-uren te trekken

        net_cost = cp.sum(
            cp.multiply(self.p_grid, self.P_prices)
            - cp.multiply(self.p_export, self.P_export_prices)
            - (self.p_solar_self * solar_bonus_rate)
        ) * self.dt

        comfort = cp.sum(
            self.s_room_low * self.P_cost_room_under +
            self.s_room_high * self.P_cost_room_over +
            self.s_dhw_low * self.P_cost_dhw_under +
            cp.multiply(cp.pos(self.s_room_low - 0.25), self.P_strictness)
        )

        valve_switches = (cp.pos(self.ufh_on[0] - self.P_init_ufh) + cp.sum(cp.pos(self.ufh_on[1:] - self.ufh_on[:-1])) +
                          cp.pos(self.dhw_on[0] - self.P_init_dhw) + cp.sum(cp.pos(self.dhw_on[1:] - self.dhw_on[:-1])))

        # Sterke straf op compressor start, lichte straf op 3-wegklep wissels
        switching_penalty = (cp.sum(self.comp_start) * 2.0) + (valve_switches * 0.2)

        stored_heat_value = (self.t_dhw[T] * self.P_val_terminal_dhw) + (self.t_room[T] * self.P_val_terminal_room)

        self.problem = cp.Problem(cp.Minimize(net_cost + comfort + switching_penalty - stored_heat_value), constraints)

    def _calc_phys(self, t_out, t_tank_or_room, mode_val):
        """Berekent natuurkunde o.b.v. GELEERDE limieten en hydraulica (ZONDER magic numbers)."""
        # 1. Haal geleerde parameters op
        lift = self.hydraulic.learned_lift_dhw if mode_val == 2 else self.hydraulic.learned_lift_ufh
        factor = self.hydraulic.learned_factor_dhw if mode_val == 2 else self.hydraulic.learned_factor_ufh

        # 2. Bepaal de start-aanvoertemperatuur o.b.v. de geleerde hydraulische curves
        if mode_val == 2:  # DHW
            # De opwarming (delta_t) is de geleerde basis + slope-correctie voor buiten
            delta_t_learned = self.hydraulic.dhw_delta_base + (self.hydraulic.dhw_delta_slope * t_out)
            t_sup_start = t_tank_or_room + lift + delta_t_learned
        else:  # UFH
            # De opwarming volgt de geleerde stooklijn
            t_sup_start = t_tank_or_room + lift + self.hydraulic.get_ufh_slope(t_out)

        # 3. Bepaal elektrisch vermogen van de compressor
        if mode_val == 2:
            # Voor DHW: Geleerde compressor-limiet (Non-modulating)
            p_el = self.perf_map.get_max_pel(t_out, t_tank_or_room)
        else:
            # Voor UFH: Bereken welk vermogen de vloer kan opnemen bij de stooklijn-temperatuur
            # Formule: P_th = K_emit * (T_sup - T_room) / (1 + K/2f)
            num = self.ident.K_emit * (t_sup_start - t_tank_or_room)
            den = 1 + (self.ident.K_emit / (2 * factor))
            p_th_emit = max(0.0, num / den)
            est_cop = self.perf_map.predict_cop(t_out, t_sup_start, mode_val)
            p_el = p_th_emit / est_cop if est_cop > 0 else 0

        # 4. Verfijn de balans (Internal Balance Loop)
        # We laten de COP en T_sup op elkaar reageren tot ze stabiel zijn
        t_sup = t_sup_start
        cop = 0
        p_th = 0

        for _ in range(3):
            t_sup = min(t_sup, self.ident.T_max_dhw)
            cop = self.perf_map.predict_cop(t_out, t_sup, mode_val)
            p_th = p_el * cop
            # De nieuwe aanvoer is: Retour (Tank + Lift) + Opwarming (P_th / Flow)
            t_sup = t_tank_or_room + lift + (p_th / factor)

        t_sup = min(t_sup, self.ident.T_max_dhw)

        return p_th, p_el, t_sup, cop

    def solve(self, state, forecast_df, recent_history_df, solar_gains):
        T = self.horizon
        r_min, r_max, d_min, d_max = self._get_targets(state["now"], T)
        self.P_t_room_init.value = np.array(state["room_temp"])
        self.P_t_dhw_init.value = np.array((state["dhw_top"] + state["dhw_bottom"]) / 2.0)

        self.P_init_ufh.value = np.array(1.0 if state["hvac_mode"] == HvacMode.HEATING.value else 0.0)
        self.P_init_dhw.value = np.array(1.0 if state["hvac_mode"] == HvacMode.DHW.value else 0.0)
        was_running = 1.0 if state["hvac_mode"] in[HvacMode.HEATING.value, HvacMode.DHW.value] else 0.0
        self.P_comp_on_init.value = np.array(was_running)

        self.P_temp_out.value = forecast_df.temp.values[:T]

        prices = np.full(T, 0.22)
        self.P_prices.value = prices
        self.P_export_prices.value = np.full(T, 0.05)

        # Dynamische Cost scaling (DPP Safe)
        avg_price = max(float(np.mean(prices)), 0.10) # Minimum 10 cent om te voorkomen dat straffen wegvallen
        self.P_cost_room_under.value = 15.0 * self.ident.C * avg_price
        self.P_cost_room_over.value = 2.0 * self.ident.C * avg_price
        self.P_cost_dhw_under.value = 15.0 * self.ident.C_tank * avg_price
        self.P_val_terminal_room.value = self.ident.C * avg_price
        self.P_val_terminal_dhw.value = self.ident.C_tank * avg_price

        self.P_solar.value = forecast_df.power_corrected.values[:T]
        self.P_base_load.value = forecast_df.load_corrected.values[:T]
        self.P_room_min.value, self.P_room_max.value = r_min, r_max
        self.P_dhw_min.value, self.P_dhw_max.value = d_min, d_max
        self.P_solar_gain.value = solar_gains[:T]

        strict_factors = np.where(self.P_temp_out.value < 0, 100.0,
                         np.where(self.P_temp_out.value < 10, 50.0,
                         np.where(self.P_temp_out.value < 15, 10.0, 1.0)))
        self.P_strictness.value = strict_factors * self.ident.C * avg_price

        # --- POWER CURVE GENERATIE (Vectorized o.b.v. GELEERDE DATA) ---
        sd_el, cd_el, sd_th, cd_th = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        su_el, cu_el, su_th, cu_th = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        for t in range(T):
            to = forecast_df.temp.values[t]
            # Boiler Curve (20C en 55C als ankerpunten)
            pth_dl, pel_dl, _, _ = self._calc_phys(to, 20.0, 2)
            pth_dh, pel_dh, _, _ = self._calc_phys(to, 55.0, 2)
            sd_el[t] = (pel_dh - pel_dl) / 35.0;
            cd_el[t] = pel_dl - (sd_el[t] * 20.0)
            sd_th[t] = (pth_dh - pth_dl) / 35.0;
            cd_th[t] = pth_dl - (sd_th[t] * 20.0)

            # Vloer Curve (18C en 22C als ankerpunten)
            pth_ul, pel_ul, _, _ = self._calc_phys(to, 18.0, 1)
            pth_uh, pel_uh, _, _ = self._calc_phys(to, 22.0, 1)
            su_el[t] = (pel_uh - pel_ul) / 4.0;
            cu_el[t] = pel_ul - (su_el[t] * 18.0)
            su_th[t] = (pth_uh - pth_ul) / 4.0;
            cu_th[t] = pth_ul - (su_th[t] * 18.0)

        self.P_dhw_pel_slope.value = sd_el;
        self.P_dhw_pel_const.value = cd_el
        self.P_dhw_pth_slope.value = sd_th;
        self.P_dhw_pth_const.value = cd_th
        self.P_ufh_pel_slope.value = su_el;
        self.P_ufh_pel_const.value = cu_el
        self.P_ufh_pth_slope.value = su_th;
        self.P_ufh_pth_const.value = cu_th

        # Historie
        lag = self.ident.ufh_lag_steps;
        hh = np.zeros(lag)
        if not recent_history_df.empty and "wp_output" in recent_history_df.columns:
            vals = recent_history_df["wp_output"].tail(lag).values
            if len(vals) > 0: hh[-len(vals):] = vals
        self.P_hist_heat.value = hh

        self.problem.solve(solver=cp.HIGHS, verbose=True)

        # --- DYNAMISCHE LOGGING (GELEERD & CONSISTENT) ---
        res_t_dh, res_t_uf = self.t_dhw.value[:-1], self.t_room.value[:-1]
        res_z_dh, res_z_uf = self.z_dhw.value, self.z_ufh.value
        res_on_dh, res_on_uf = self.dhw_on.value, self.ufh_on.value

        # P_th berekenen zoals de solver het deed
        p_th_dh_solver = (sd_th * res_z_dh + cd_th * res_on_dh)
        p_th_uf_solver = (su_th * res_z_uf + cu_th * res_on_uf)

        self.d_cop = np.divide(p_th_dh_solver, self.p_el_dhw.value, out=np.full(T, 2.5), where=self.p_el_dhw.value > 0.01)
        self.u_cop = np.divide(p_th_uf_solver, self.p_el_ufh.value, out=np.full(T, 3.5), where=self.p_el_ufh.value > 0.01)

        self.d_sup = np.where(res_on_dh > 0.5, res_t_dh + self.hydraulic.learned_lift_dhw + (p_th_dh_solver / self.hydraulic.learned_factor_dhw), 0.0)
        self.u_sup = np.where(res_on_uf > 0.5, res_t_uf + self.hydraulic.learned_lift_ufh + (p_th_uf_solver / self.hydraulic.learned_factor_ufh), 0.0)
        self.d_sup = np.minimum(self.d_sup, self.ident.T_max_dhw)

    def _get_targets(self, now, T):
        r_min, r_max, d_min, d_max = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        local_tz = datetime.now().astimezone().tzinfo
        now_local = now.astimezone(local_tz)
        for t in range(T):
            fut_time = now_local + timedelta(hours=t * self.dt)
            h = fut_time.hour
            if 17 <= h < 22:
                r_min[t], r_max[t] = 20.0, 21.5
            elif 11 <= h < 17:
                r_min[t], r_max[t] = 19.5, 22.0
            else:
                r_min[t], r_max[t] = 19.0, 19.5
            if 11 <= h <= 15:
                d_min[t] = 50.0
            else:
                d_min[t] = 10.0
            d_max[t] = 55.0
        return r_min, r_max, d_min, d_max

# =========================================================
# 5. OPTIMIZER (Met fysieke Supply Temp bepaling)
# =========================================================
class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.hp_model_path)
        self.ident = SystemIdentificator(config.rc_model_path)
        self.hydraulic = HydraulicPredictor(config.hydraulic_model_path)
        self.res_ufh = MLResidualPredictor(
            config.ufh_model_path,
            self.ident.R,
            self.ident.C,
            self.ident.K_loss_dhw,
            self.ident.C_tank,
        )
        self.res_dhw = MLResidualPredictor(
            config.dhw_model_path,
            self.ident.R,
            self.ident.C,
            self.ident.K_loss_dhw,
            self.ident.C_tank,
        )

        # Geef de hydraulic predictor mee aan MPC
        self.mpc = ThermalMPC(self.ident, self.perf_map, self.hydraulic)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)
        df = self.db.get_history(cutoff_date=cutoff)
        if df.empty:
            return

        self.perf_map.train(df)
        self.ident.train(df)
        self.hydraulic.train(df)
        # self.res_ufh.train(df)
        # self.res_dhw.train(df, is_dhw=True)
        self.mpc._build_problem()

    def resolve(self, context: Context):
        state = {
            "now": context.now,
            "hvac_mode": context.hvac_mode.value,
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        cutoff = context.now - timedelta(hours=4)
        raw_hist = self.db.get_history(cutoff_date=cutoff)

        # BUGFIX: Calculate wp_output before passing to MPC so lag works properly!
        recent_history_df = pd.DataFrame()
        if not raw_hist.empty:
            raw_hist = raw_hist.copy()
            raw_hist.set_index("timestamp", inplace=True)
            recent_history_df = (
                raw_hist.resample("15min")
                .mean(numeric_only=True)
                .fillna(0)
                .reset_index()
            )

        # FIX: Voorspel de zon-opwarming vooraf via het getrainde ML model
        solar_gains = self.res_ufh.predict(context.forecast_df)

        # Geef de solar_gains ook mee aan de solver
        self.mpc.solve(state, context.forecast_df, recent_history_df, solar_gains)

        if self.mpc.p_el_ufh.value is None:
            return {
                "mode": "OFF",
                "target_pel_kw": 0.0,
                "target_supply_temp": 0.0,
                "plan": [],
            }

        # Wat doen we NU (index 0)
        p_el_ufh_now = self.mpc.p_el_ufh.value[0]
        p_el_dhw_now = self.mpc.p_el_dhw.value[0]

        mode = "OFF"
        target_pel = 0.0
        target_supply_temp = 0.0

        # --- BEPALEN TARGETS MET ML (GEEN VASTE FORMULES MEER) ---
        if p_el_dhw_now > 0.1:
            mode = "DHW"
            target_pel = p_el_dhw_now
            target_supply_temp = self.mpc.d_sup[0]

        elif p_el_ufh_now > 0.1:
            mode = "UFH"
            target_pel = p_el_ufh_now
            target_supply_temp = self.mpc.u_sup[0]

        return {
            "mode": mode,
            "target_pel_kw": round(target_pel, 2),
            "target_supply_temp": round(target_supply_temp, 1),
            "status": self.mpc.problem.status,
            "plan": self.get_plan(context),
        }

    def get_plan(self, context):
        if self.mpc.p_el_ufh.value is None:
            return []

        plan = []
        T = self.mpc.horizon

        local_tz = datetime.now().astimezone().tzinfo
        now_local = context.now.astimezone(local_tz)

        minute = (now_local.minute // 15) * 15
        start_time = now_local.replace(minute=minute, second=0, microsecond=0)

        u_on = self.mpc.ufh_on.value
        d_on = self.mpc.dhw_on.value
        p_u = self.mpc.p_el_ufh.value
        p_d = self.mpc.p_el_dhw.value
        t_r = self.mpc.t_room.value
        t_d = self.mpc.t_dhw.value
        u_cop = self.mpc.u_cop
        d_cop = self.mpc.d_cop
        u_sup = self.mpc.u_sup
        d_sup = self.mpc.d_sup
        strictness = self.mpc.P_strictness.value

        for t in range(T):
            ts = start_time + timedelta(minutes=t * 15)
            mode_str = "-"
            if d_on[t] > 0.5:
                mode_str = "DHW"
            elif u_on[t] > 0.5:
                mode_str = "UFH"

            plan.append(
                {
                    "time": ts,
                    "mode": mode_str,
                    "t_out": f"{context.forecast_df.temp.iloc[t]:.1f}",
                    "p_solar": f"{context.forecast_df.power_corrected.iloc[t]:.2f}",
                    "p_load": f"{context.forecast_df.load_corrected.iloc[t]:.2f}",
                    "t_room": f"{t_r[t + 1]:.1f}",
                    "t_dhw": f"{t_d[t + 1]:.1f}",
                    "p_el_ufh": f"{p_u[t]:.2f}",
                    "p_el_dhw": f"{p_d[t]:.2f}",
                    "cop_ufh": f"{u_cop[t]:.2f}",
                    "cop_dhw": f"{d_cop[t]:.2f}",
                    "supply_ufh": f"{u_sup[t]:.2f}",
                    "supply_dhw": f"{d_sup[t]:.2f}",
                    "strictness": f"{strictness[t]:.0f}",
                }
            )

        return plan
