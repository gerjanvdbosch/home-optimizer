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
        self.power_model = None
        self.max_freq_model = None

        # Default referenties
        self.ufh_freq_ref = 35.0
        self.dhw_freq_ref = 60.0
        self.ufh_delta_t_ref = 4.0
        self.dhw_delta_t_ref = 7.0

        self.features_cop = [
            "temp",
            "supply_temp",
            "return_temp",
            "delta_t",
            "compressor_freq",
            "hvac_mode",
        ]
        self.features_power = ["compressor_freq", "temp", "supply_temp", "hvac_mode"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.cop_model = data["cop_model"]
                self.power_model = data["power_model"]
                self.max_freq_model = data.get("max_freq_model")
                self.ufh_freq_ref = data.get("ufh_freq_ref", 35.0)
                self.dhw_freq_ref = data.get("dhw_freq_ref", 60.0)
                self.ufh_delta_t_ref = data.get("ufh_delta_t_ref", 4.0)
                self.dhw_delta_t_ref = data.get("dhw_delta_t_ref", 7.0)
                self.is_fitted = True
                logger.info(
                    f"[Optimizer] Loaded refs: UFH={self.ufh_freq_ref:.1f}Hz/dT={self.ufh_delta_t_ref:.1f}, "
                    f"DHW={self.dhw_freq_ref:.1f}Hz/dT={self.dhw_delta_t_ref:.1f}"
                )
            except Exception as e:
                logger.warning(f"[Optimizer] Performance map laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        """
        Train de modellen met fysiek correcte flow-berekeningen.
        """
        df = df.copy()

        # ============================
        # 1. Fysisch thermisch vermogen berekenen
        # ============================
        df["delta_t"] = (df["supply_temp"] - df["return_temp"]).astype(float)

        # Verwijder ongeldige situaties (defrost / meetfouten / stilstand)
        df = df[df["delta_t"] > 0.5]

        # Bereken Thermisch Vermogen (wp_output) afhankelijk van modus
        # Gebruik numpy select voor snelheid en correctheid
        conditions = [
            df["hvac_mode"] == HvacMode.HEATING.value,
            df["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices = [
            df["delta_t"] * FACTOR_UFH,  # 18 L/min
            df["delta_t"] * FACTOR_DHW,  # 19 L/min
        ]

        # Default 0 als het geen heating/dhw is (zou gefilterd moeten worden)
        df["wp_output"] = np.select(conditions, choices, default=0.0)

        # Verwijder regels waar wp_output 0 is (verkeerde modus) of wp_actual te laag (standby)
        df = df[(df["wp_output"] > 0.1) & (df["wp_actual"] > 0.2)].copy()

        # COP berekenen: P_thermisch / P_elektrisch
        df["cop"] = df["wp_output"] / df["wp_actual"]

        # Fysische filtering (outlier removal)
        mask = (
            (df["compressor_freq"] > 15)
            & (df["cop"].between(0.8, 8.0))
            & (df["delta_t"] > 1.0)
            & (df["delta_t"] < 15.0)
        )
        df_clean = df[mask].copy()

        if len(df_clean) < 50:
            logger.warning(
                "[Optimizer] Te weinig valide data voor performance map training."
            )
            return

        # ============================
        # 2. COP model (Random Forest)
        # ============================
        self.cop_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        self.cop_model.fit(df_clean[self.features_cop], df_clean["cop"])

        # ============================
        # 3. Elektrisch vermogen model (Lineair)
        # ============================
        # We fitten P_el op frequentie en temperaturen voor een basislijn
        self.power_model = LinearRegression(fit_intercept=False)
        self.power_model.fit(df_clean[self.features_power], df_clean["wp_actual"])

        # ============================
        # 4. Max frequentie vs buitentemp
        # ============================
        df_f = df_clean.copy()
        df_f["t_rounded"] = df_f["temp"].round()

        max_freq_stats = (
            df_f.groupby("t_rounded")["compressor_freq"].quantile(0.98).reset_index()
        )

        if not max_freq_stats.empty:
            self.max_freq_model = LinearRegression().fit(
                max_freq_stats[["t_rounded"]],
                max_freq_stats["compressor_freq"],
            )

        # ============================
        # 5. Update Referentie Frequenties en Delta T's
        # ============================
        mask_ufh = df_clean["hvac_mode"] == HvacMode.HEATING.value
        if mask_ufh.any():
            self.ufh_freq_ref = float(
                df_clean.loc[mask_ufh, "compressor_freq"].median()
            )
            self.ufh_delta_t_ref = float(
                df_clean.loc[mask_ufh, "delta_t"].clip(2, 8).median()
            )

        mask_dhw = df_clean["hvac_mode"] == HvacMode.DHW.value
        if mask_dhw.any():
            self.dhw_freq_ref = float(
                df_clean.loc[mask_dhw, "compressor_freq"].median()
            )
            self.dhw_delta_t_ref = float(
                df_clean.loc[mask_dhw, "delta_t"].clip(4, 10).median()
            )

        logger.info(
            f"[Optimizer] Trained. Refs: UFH={self.ufh_freq_ref:.1f}Hz/dT={self.ufh_delta_t_ref:.1f}, "
            f"DHW={self.dhw_freq_ref:.1f}Hz/dT={self.dhw_delta_t_ref:.1f}"
        )

        self.is_fitted = True

        joblib.dump(
            {
                "cop_model": self.cop_model,
                "power_model": self.power_model,
                "max_freq_model": self.max_freq_model,
                "ufh_freq_ref": self.ufh_freq_ref,
                "dhw_freq_ref": self.dhw_freq_ref,
                "ufh_delta_t_ref": self.ufh_delta_t_ref,
                "dhw_delta_t_ref": self.dhw_delta_t_ref,
            },
            self.path,
        )

    def predict_cop(self, t_out, t_supply, t_return, freq, mode_idx):
        if not self.is_fitted:
            return 3.5 if mode_idx == HvacMode.HEATING.value else 2.5

        delta_t = max(0.5, t_supply - t_return)
        data = pd.DataFrame(
            [[t_out, t_supply, t_return, delta_t, freq, mode_idx]],
            columns=self.features_cop,
        )
        return float(self.cop_model.predict(data)[0])

    def predict_p_el_slope(self, freq_ref, t_out, t_sink, mode_idx):
        """
        Voorspelt elektrisch vermogen per Hz (slope) rondom het referentiepunt.
        """
        if not self.is_fitted:
            return 0.04  # Default fallback

        data = pd.DataFrame(
            [[freq_ref, t_out, t_sink, mode_idx]], columns=self.features_power
        )
        p_el_pred = self.power_model.predict(data)[0]

        # Slope = Power / Freq. Zorg dat we niet door 0 delen of negatief gaan.
        return max(0.01, p_el_pred / max(10.0, freq_ref))

    def predict_max_freq(self, t_out):
        if self.max_freq_model is None:
            return 75.0

        t_rounded = round(t_out)
        pred = self.max_freq_model.predict(
            pd.DataFrame([[t_rounded]], columns=["t_rounded"])
        )[0]
        return float(np.clip(pred, 30.0, 90.0))


# =========================================================
# 2. SYSTEM IDENTIFICATOR
# =========================================================
class SystemIdentificator:
    def __init__(self, path):
        self.path = Path(path)
        self.R = 15.0  # K/kW
        self.C = 30.0  # kWh/K
        self.K_emit = 0.15  # kW/K (Afgifte vloer)
        self.K_tank = 0.25  # kW/K (Afgifte spiraal)
        self.K_loss_dhw = 0.15  # °C/uur verlies
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.R = data.get("R", 15.0)
                self.C = data.get("C", 30.0)
                self.K_emit = data.get("K_emit", 0.15)
                self.K_tank = data.get("K_tank", 0.25)
                self.K_loss_dhw = data.get("K_loss_dhw", 0.15)
                logger.info(
                    f"[Optimizer] Loaded system ID: R={self.R:.1f}K/W, C={self.C:.1f}kWh/K, "
                    f"K_emit={self.K_emit:.3f}kW/°C, K_tank={self.K_tank:.3f}kW/°C, K_loss={self.K_loss_dhw:.3f}°C/h"
                )
            except Exception as e:
                logger.error(f"Failed to load SystemID: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy().sort_values("timestamp")

        # ============================
        # 0. Data Voorbereiding (Algemeen)
        # ============================
        # Zorg dat alle kolommen numeriek zijn
        cols_to_numeric = [
            "supply_temp",
            "return_temp",
            "room_temp",
            "temp",
            "dhw_top",
            "dhw_bottom",
            "pv_actual",
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["delta_t"] = (df["supply_temp"] - df["return_temp"]).clip(lower=0.0)

        # Bereken output op basis van HVAC mode
        conditions = [
            df["hvac_mode"] == HvacMode.HEATING.value,
            df["hvac_mode"] == HvacMode.DHW.value,
        ]
        choices = [df["delta_t"] * FACTOR_UFH, df["delta_t"] * FACTOR_DHW]
        df["wp_output"] = np.select(conditions, choices, default=0.0)

        # ============================
        # Dataset A: Uurgemiddelden (Voor R & C)
        # ============================
        # Traagheid van het gebouw zie je pas na uren.
        df_1h = (
            df.set_index("timestamp")
            .resample("1h")
            .agg(
                {
                    "room_temp": "mean",
                    "temp": "mean",
                    "wp_output": "mean",
                    "pv_actual": "mean",
                    "hvac_mode": lambda x: x.mode()[0] if not x.mode().empty else 0,
                }
            )
            .dropna()
            .reset_index()
        )

        df_1h["dT_next"] = df_1h["room_temp"].shift(-1) - df_1h["room_temp"]
        df_1h["delta_T_env"] = df_1h["room_temp"] - df_1h["temp"]
        df_1h = df_1h.dropna()

        # ============================
        # Dataset B: Kwartierwaarden (Voor K_emit & K_tank)
        # ============================
        # Afgifte van vloer/spiraal is een snel proces, 15min data is hier beter.
        df_15m = (
            df.set_index("timestamp").resample("15min").mean().dropna().reset_index()
        )

        # ==========================================
        # STAP 1: Afkoeling (Bepaal Tau = R * C)
        # ==========================================
        # Filter: Nacht, WP uit, Temperatuur daalt, Buiten koud genoeg
        mask_cooling = (
            (df_1h["wp_output"] < 0.1)  # WP uit
            & (df_1h["pv_actual"] < 0.1)  # Nacht
            & (df_1h["dT_next"] < -0.01)  # Temp daalt significant
            & (df_1h["delta_T_env"] > 4.0)  # Binnen warmer dan buiten
        )
        df_cool = df_1h[mask_cooling].copy()

        tau_inv = None  # Dit is 1 / (R * C)

        if len(df_cool) > 10:
            print("Debug Cooling Data Sample:")
            print(
                df_cool[
                    ["timestamp", "room_temp", "temp", "dT_next", "delta_T_env"]
                ].head(30)
            )

            # Model: dT/dt = -1/(RC) * (T_room - T_out) -> Y = a * X
            X_cool = -df_cool[["delta_T_env"]]
            y_cool = df_cool["dT_next"]

            lr_cool = LinearRegression(fit_intercept=False)
            lr_cool.fit(X_cool, y_cool)

            tau_inv = lr_cool.coef_[0]

            # Sanity Check: Tau tussen 50 uur (slecht) en 1000 uur (passiefhuis)
            # Tau_inv dus tussen 0.02 en 0.001
            if tau_inv < 0.0005 or tau_inv > 0.05:
                logger.warning(
                    f"[SysID] Cooling fit onbetrouwbaar (Tau_inv={tau_inv:.5f})."
                )
                tau_inv = None
            else:
                logger.info(
                    f"[SysID] Cooling identified: Tau (RC) = {1 / tau_inv:.1f} uur"
                )

        # ==========================================
        # STAP 2: Verwarming (Bepaal C)
        # ==========================================
        # Fallback voor Tau als afkoeling niet gemeten kon worden
        if tau_inv is None:
            tau_inv = 1.0 / (self.R * self.C)

        # Filter: WP aan, Temp stijgt
        mask_heat = (df_1h["wp_output"] > 0.7) & (df_1h["dT_next"] > 0.01)
        df_heat = df_1h[mask_heat].copy()

        if len(df_heat) > 10:
            print("Debug Heating Data Sample:")
            print(
                df_heat[
                    ["timestamp", "room_temp", "temp", "dT_next", "delta_T_env"]
                ].head(30)
            )

            # Model: dT/dt + (1/RC)*dT_env = (1/C)*P
            y_heat_adjusted = df_heat["dT_next"] + (tau_inv * df_heat["delta_T_env"])
            X_heat = df_heat[["wp_output"]]

            lr_heat = LinearRegression(fit_intercept=False)
            lr_heat.fit(X_heat, y_heat_adjusted)

            inv_C = lr_heat.coef_[0]  # 1/C

            if inv_C > 0.001:
                new_C = 1.0 / inv_C
                # R = Tau / C
                new_R = (1.0 / tau_inv) / new_C

                if 10.0 < new_C < 200.0 and 2.0 < new_R < 60.0:
                    self.C = float(new_C)
                    self.R = float(new_R)
                    logger.info(
                        f"[SysID] SPLIT TRAINED: R={self.R:.1f} K/kW, C={self.C:.1f} kWh/K"
                    )
                else:
                    logger.warning(
                        f"[SysID] Waarden buiten bereik: R={new_R:.1f}, C={new_C:.1f}"
                    )

        # ============================
        # 3. Identificatie K_emit (Vloerafgifte)
        # ============================
        mask_ufh = (
            (df_15m["hvac_mode"] == HvacMode.HEATING.value)
            & (df_15m["wp_output"] > 0.7)
            & (df_15m["supply_temp"] < 35)  # Niet te heet (voorkom overshoot data)
            & (df_15m["supply_temp"] > df_15m["room_temp"] + 2)  # Zinvolle delta T
        )
        df_ufh = df_15m[mask_ufh].copy()

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
            (df_15m["hvac_mode"] == HvacMode.DHW.value)
            & (df_15m["wp_output"] > 1.7)
            & (df_15m["supply_temp"] > 35)
        )
        df_dhw = df_15m[mask_dhw].copy()

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
            t_tank = (df_dhw["dhw_top"] + df_dhw["dhw_bottom"]) / 2
            t_avg_water = (df_dhw["supply_temp"] + df_dhw["return_temp"]) / 2
            delta_T_hx = t_avg_water - t_tank

            # Alleen als water warmer is dan tank
            valid_idx = delta_T_hx > 2.0
            if valid_idx.any():
                k_values = df_dhw.loc[valid_idx, "wp_output"] / delta_T_hx[valid_idx]
                self.K_tank = float(np.clip(k_values.median(), 0.15, 2.0))

        # ============================
        # 5. Identificatie Stilstandsverlies DHW
        # ============================
        # Gebruik originele DF voor tijdstappen, want resample kan gaten maskeren
        df_loss = df.sort_values("timestamp").copy()
        df_loss["t_tank"] = (df_loss["dhw_top"] + df_loss["dhw_bottom"]) / 2.0

        # FIX: Gebruik de kolom timestamp voor de diff, niet de index
        df_loss["dt_hours"] = df_loss["timestamp"].diff().dt.total_seconds() / 3600.0
        df_loss["dT_tank"] = df_loss["t_tank"].diff()

        mask_sb = (
            (df_loss["hvac_mode"] != HvacMode.DHW.value)
            & (df_loss["wp_output"] < 0.1)
            & (df_loss["dT_tank"] < 0)
            & (df_loss["dt_hours"] > 0.1)
            & (df_loss["dt_hours"] < 1.0)
        )

        df_sb = df_loss[mask_sb].copy()
        if len(df_sb) > 20:
            print("Debug Stilstandsverlies DHW Sample:")
            print(df_sb[["timestamp", "dT_tank", "dt_hours", "wp_output"]].head(30))

            # Loss rate positief maken voor berekening
            loss_rates = -(df_sb["dT_tank"] / df_sb["dt_hours"])

            # Filter: Verlies > 0.5 °C/uur is bijna altijd tappen, geen stilstand
            loss_rates = loss_rates[loss_rates < 0.5]

            if len(loss_rates) > 5:
                # Gebruik het 25e percentiel om 'klein gebruik' eruit te filteren
                # en de echte isolatiewaarde te vinden.
                self.K_loss_dhw = float(np.clip(loss_rates.quantile(0.25), 0.02, 0.4))

        logger.info(
            f"[Optimizer] Final Trained SysID: R={self.R:.1f}, C={self.C:.1f}, "
            f"K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f}, K_loss={self.K_loss_dhw:.3f}"
        )

        joblib.dump(
            {
                "R": self.R,
                "C": self.C,
                "K_emit": self.K_emit,
                "K_tank": self.K_tank,
                "K_loss_dhw": self.K_loss_dhw,
            },
            self.path,
        )


# =========================================================
# 3. ML RESIDUALS
# =========================================================
class MLResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        # Alleen externe factoren en tijd.
        # wp_output en room_temp halen we eruit voor de MPC-stabiliteit.
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

            # Fysische basis inclusief het vaste K_loss uit je SysID
            t_model_next = t_curr + (p_heat * dt / tank_cap) - (self.K_loss_dhw * dt)

        # Residu berekenen (K/u)
        target = (t_next - t_model_next) / dt

        df_feat = add_cyclic_time_features(df_feat, "timestamp")
        df_feat["solar"] = df_feat["pv_actual"]
        df_feat["target"] = target

        train_set = df_feat[self.features + ["target"]].dropna()

        # Filter uitschieters (bijv. douchen is -10 K/u, dat moet de ML niet willen leren)
        if is_dhw:
            train_set = train_set[train_set["target"].between(-4.0, 0.5)]
            print("Debug ML Residuals Sample (DHW):")
        else:
            train_set = train_set[train_set["target"].between(-0.5, 0.5)]
            print("Debug ML Residuals Sample (UFH):")

        print(train_set.head(30))

        if len(train_set) > 10:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=15
            ).fit(train_set[self.features], train_set["target"])
            joblib.dump(self.model, self.path)

    def predict(self, forecast_df):
        if self.model is None:
            return np.zeros(len(forecast_df))

        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        df["solar"] = df["power_corrected"]

        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0

        return self.model.predict(df[self.features])


# =========================================================
# 4. THERMAL MPC
# =========================================================
class ThermalMPC:
    def __init__(self, ident, perf_map):
        self.ident = ident
        self.perf_map = perf_map
        self.horizon, self.dt = 48, 0.25
        self.room_target = 20.0
        self.dhw_target = 50.0
        self._build_problem()

    def _build_problem(self):
        T = self.horizon

        self.P_t_room_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()
        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_export_prices = cp.Parameter(T, nonneg=True)
        self.P_temp_out = cp.Parameter(T)

        self.P_max_freq = cp.Parameter(T, nonneg=True)
        self.P_dhw_loss_per_dt = cp.Parameter(nonneg=True)

        self.P_th_per_hz_ufh = cp.Parameter(T, nonneg=True)
        self.P_th_per_hz_dhw = cp.Parameter(T, nonneg=True)
        self.P_el_per_hz_ufh = cp.Parameter(T, nonneg=True)
        self.P_el_per_hz_dhw = cp.Parameter(T, nonneg=True)

        self.P_ufh_res = cp.Parameter(T)
        self.P_dhw_res = cp.Parameter(T)
        self.P_solar = cp.Parameter(T, nonneg=True)
        self.P_base_load = cp.Parameter(T, nonneg=True)

        self.f_ufh = cp.Variable(T, nonneg=True)
        self.f_dhw = cp.Variable(T, nonneg=True)
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.p_grid = cp.Variable(T, nonneg=True)
        self.p_export = cp.Variable(T, nonneg=True)
        self.p_solar_self = cp.Variable(T, nonneg=True)
        self.t_room = cp.Variable(T + 1)
        self.t_dhw = cp.Variable(T + 1)
        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)

        R, C = self.ident.R, self.ident.C

        constraints = [
            self.t_room[0] == self.P_t_room_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        for t in range(T):
            # 1. Definieer vermogens
            p_el_wp = (
                self.f_ufh[t] * self.P_el_per_hz_ufh[t]
                + self.f_dhw[t] * self.P_el_per_hz_dhw[t]
            )
            # FIX: Thermisch vermogen moet komen uit frequentie x thermisch-per-Hz
            p_th_ufh = self.f_ufh[t] * self.P_th_per_hz_ufh[t]
            p_th_dhw = self.f_dhw[t] * self.P_th_per_hz_dhw[t]

            total_load = p_el_wp + self.P_base_load[t]

            # 2. Elektrische Balans & PV Balans
            constraints += [total_load == self.p_grid[t] + self.p_solar_self[t]]
            constraints += [self.P_solar[t] == self.p_solar_self[t] + self.p_export[t]]

            # 3. Thermische Balans (R-C en Tank model)
            constraints += [
                # FIX: Gebruik p_th_ufh
                self.t_room[t + 1]
                == self.t_room[t]
                + (
                    (p_th_ufh - (self.t_room[t] - self.P_temp_out[t]) / R) * self.dt / C
                    + (self.P_ufh_res[t] * self.dt)
                ),
                # FIX: Gebruik p_th_dhw
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (
                    (p_th_dhw * self.dt) / 0.232
                    + (self.P_dhw_res[t] * self.dt)
                    - self.P_dhw_loss_per_dt
                ),
                # Machine & Veiligheidsgrenzen
                self.f_ufh[t] <= self.ufh_on[t] * self.P_max_freq[t],
                self.f_dhw[t] <= self.dhw_on[t] * self.P_max_freq[t],
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                self.f_ufh[t] >= self.ufh_on[t] * 25,
                self.f_dhw[t] >= self.dhw_on[t] * 35,
                self.t_room[t + 1] + self.s_room_low[t] >= 18.0,
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= 25.0,
                self.t_room[t + 1] <= 22.5,
            ]

        # --- OBJECTIVE FUNCTION (Onveranderd) ---
        net_cost = (
            cp.sum(
                cp.multiply(self.p_grid, self.P_prices)
                - cp.multiply(self.p_export, self.P_export_prices)
            )
            * self.dt
        )
        comfort_room_low = cp.sum(cp.pos(self.room_target - self.t_room)) * 4.0
        comfort_dhw_low = cp.sum(cp.pos(self.dhw_target - self.t_dhw)) * 2.0

        # NIEUW: Straf voor te warm (pos(T - Target)) -> Dit dwingt uitschakeling af!
        comfort_room_high = cp.sum(cp.pos(self.t_room - 21.0)) * 5.0
        comfort_dhw_high = cp.sum(cp.pos(self.t_dhw - 51.0)) * 2.0

        # 4. Schakelkosten (Switching Costs)
        # Verlaagd van 10.0 naar 0.5 om uitschakelen rendabel te maken
        ufh_switch = cp.sum(cp.abs(self.ufh_on[1:] - self.ufh_on[:-1])) * 0.5
        dhw_switch = cp.sum(cp.abs(self.dhw_on[1:] - self.dhw_on[:-1])) * 0.5

        # 5. Veiligheid (Absolute bodem)
        safety_violation = cp.sum(self.s_room_low + self.s_dhw_low) * 100

        # TOTAAL
        self.problem = cp.Problem(
            cp.Minimize(
                net_cost
                + comfort_room_low
                + comfort_room_high
                + comfort_dhw_low
                + comfort_dhw_high
                + ufh_switch
                + dhw_switch
                + safety_violation
            ),
            constraints,
        )

    def solve(self, state, forecast_df, res_u, res_d):
        T = self.horizon
        t_out = forecast_df.temp.values[:T]
        t_prices = [0.22] * T
        t_export_prices = [0.05] * T
        t_solar = forecast_df.power_corrected.values[:T]
        t_base_load = forecast_df.load_corrected.values[:T]

        res_u = res_u[:T]
        res_d = res_d[:T]

        dhw_start = (state["dhw_top"] + state["dhw_bottom"]) / 2
        current_est_room, current_est_dhw = state["room_temp"], dhw_start

        # Tijdelijke opslag voor gecombineerde waarden (DPP optimalisatie)
        th_per_hz_u, th_per_hz_d = np.zeros(T), np.zeros(T)
        el_per_hz_u, el_per_hz_d = np.zeros(T), np.zeros(T)
        v_max_freq = np.zeros(T)

        # Haal alle geleerde parameters op
        ufh_ref = self.perf_map.ufh_freq_ref
        dhw_ref = self.perf_map.dhw_freq_ref
        ufh_dt = self.perf_map.ufh_delta_t_ref
        dhw_dt = self.perf_map.dhw_delta_t_ref

        # Stel stilstandsverlies parameter in
        self.P_dhw_loss_per_dt.value = self.ident.K_loss_dhw * self.dt

        for t in range(T):
            # Schat de maximale frequentie voor dit tijdstip
            v_max_freq[t] = self.perf_map.predict_max_freq(t_out[t])

            # 1. UFH: Dynamische Stooklijn o.b.v. warmteverlies huis
            calc_room = max(current_est_room, 20.0)
            heat_loss_kw = max(0, (calc_room - t_out[t]) / self.ident.R)
            # Overtemp = Vermogen / Afgiftecoëfficiënt
            overtemp_ufh = np.clip(heat_loss_kw / self.ident.K_emit, 0, 15.0)
            t_supply_ufh = calc_room + (ufh_dt / 2.0) + overtemp_ufh

            cop_u = self.perf_map.predict_cop(
                t_out[t],
                t_supply_ufh,
                t_supply_ufh - ufh_dt,
                ufh_ref,
                HvacMode.HEATING.value,
            )
            # Slope nu inclusief aanvoertemperatuur (t_sink) en modus
            slope_u = self.perf_map.predict_p_el_slope(
                ufh_ref, t_out[t], t_supply_ufh, HvacMode.HEATING.value
            )

            el_per_hz_u[t] = slope_u
            th_per_hz_u[t] = slope_u * cop_u

            # 2. DHW: Dynamische Aanvoer o.b.v. vermogen warmtepomp
            calc_dhw = max(current_est_dhw, 40.0)

            # Stap A: Maak een eerste schatting van het thermisch vermogen
            # We gebruiken hier calc_dhw + dhw_dt als tijdelijke sink voor de allereerste COP-check
            p_th_est = (
                dhw_ref
                * self.perf_map.predict_p_el_slope(
                    dhw_ref, t_out[t], calc_dhw + dhw_dt, HvacMode.DHW.value
                )
                * self.perf_map.predict_cop(
                    t_out[t], calc_dhw + dhw_dt, calc_dhw, dhw_ref, HvacMode.DHW.value
                )
            )

            # Stap B: Bereken de overtemp die nodig is om DIT vermogen door de spiraal te duwen
            # Overtemp = P_th / K_tank
            overtemp_dhw = np.clip(p_th_est / self.ident.K_tank, 0, 20.0)

            # Stap C: De aanvoer is de tank-temp + de helft van de water-delta + de overtemp over de spiraal
            t_supply_dhw = min(calc_dhw + (dhw_dt / 2.0) + overtemp_dhw, 58.0)

            # Stap D: De definitieve waarden voor de solver
            cop_d = self.perf_map.predict_cop(
                t_out[t],
                t_supply_dhw,
                t_supply_dhw - dhw_dt,
                dhw_ref,
                HvacMode.DHW.value,
            )
            slope_d = self.perf_map.predict_p_el_slope(
                dhw_ref, t_out[t], t_supply_dhw, HvacMode.DHW.value
            )

            el_per_hz_d[t] = slope_d
            th_per_hz_d[t] = slope_d * cop_d

            # 3. State Update
            current_est_room = (
                calc_room
                - (calc_room - t_out[t]) / (self.ident.R * self.ident.C) * self.dt
                + res_u[t]
            )
            current_est_dhw = calc_dhw - (self.ident.K_loss_dhw * self.dt) + res_d[t]

            if t < 16:  # 16 stappen = 4 uur
                logger.debug(
                    f"[MPC Debug] t={t*0.25:.2f}h | T_out={t_out[t]:.1f} | T_room_calc={calc_room:.1f} | "
                    f"HeatLoss={(calc_room - t_out[t]) / self.ident.R:.2f}kW | "
                    f"COP_U={cop_u:.2f} | P_el_slope={slope_u:.3f} | P_th_per_hz={th_per_hz_u[t]:.3f}"
                )

        # Vul parameters
        self.P_max_freq.value = v_max_freq
        self.P_th_per_hz_ufh.value = th_per_hz_u
        self.P_th_per_hz_dhw.value = th_per_hz_d
        self.P_el_per_hz_ufh.value = el_per_hz_u
        self.P_el_per_hz_dhw.value = el_per_hz_d

        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = dhw_start
        self.P_temp_out.value = t_out
        self.P_prices.value = t_prices
        self.P_export_prices.value = t_export_prices
        self.P_solar.value = t_solar
        self.P_base_load.value = t_base_load
        self.P_ufh_res.value = res_u
        self.P_dhw_res.value = res_d

        try:
            self.problem.solve(solver=cp.CBC)
            return self.f_ufh.value, self.f_dhw.value
        except Exception as e:
            logger.error(f"MPC Solve failed: {e}")
            return np.zeros(T), np.zeros(T)


# =========================================================
# 5. OPTIMIZER
# =========================================================
class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.hp_model_path)
        self.ident = SystemIdentificator(config.rc_model_path)
        self.res_ufh = MLResidualPredictor(config.ufh_model_path)
        self.res_dhw = MLResidualPredictor(config.dhw_model_path)
        self.mpc = ThermalMPC(self.ident, self.perf_map)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)

        df = self.db.get_history(cutoff_date=cutoff)
        if df.empty:
            return

        self.perf_map.train(df)
        self.ident.train(df)
        self.res_ufh.train(df, self.ident.R, self.ident.C)
        self.res_dhw.train(df, self.ident.R, self.ident.C, True, self.ident.K_loss_dhw)

    def resolve(self, context: Context):
        res_u, res_d = self.res_ufh.predict(context.forecast_df), self.res_dhw.predict(
            context.forecast_df
        )

        state = {
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        f_ufh, f_dhw = self.mpc.solve(state, context.forecast_df, res_u, res_d)

        hz = f_ufh[0] if f_ufh[0] > 15 else f_dhw[0] if f_dhw[0] > 15 else 0
        mode = "UFH" if f_ufh[0] > 15 else "DHW" if f_dhw[0] > 15 else "OFF"

        return {"mode": mode, "freq": round(hz, 1)}
