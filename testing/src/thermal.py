"""
thermal.py

Bevat:
  - clean_thermal_data()        — filtert fysisch corrupte rijen
  - HPPerformanceMap            — leert P_el en COP direct uit data
  - SystemIdentificator         — leert R, C, K_emit, K_tank, lag
  - HydraulicPredictor          — leert supply-temp en hydraulische parameters
  - UfhResidualPredictor        — leert zonopwarming en overige residuen
  - DhwResidualPredictor        — leert warm-water vraagpatroon
  - PhysicsLinearizer           — SLP-linearisatie met convergentiecheck
  - ComfortCostCalculator       — dimensioneel correcte comfortboetes
"""

import logging
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from model import ModelSelector
from utils import add_cyclic_time_features
from context import HvacMode

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTEN — alleen als fysische fallback, nooit als aanname
# =========================================================
_FLOW_UFH = 18.0  # L/min
_FLOW_DHW = 19.0
_CP_WATER = 4.186  # kJ/kg·K

FACTOR_UFH = (_FLOW_UFH / 60.0) * _CP_WATER  # ~1.256 kW/K
FACTOR_DHW = (_FLOW_DHW / 60.0) * _CP_WATER  # ~1.326 kW/K


# =========================================================
# DATA CLEANING
# =========================================================


def clean_thermal_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verwijdert fysisch corrupte rijen. Gooit alleen weg wat aantoonbaar
    fout is — geen schattingen, geen imputatie.

    Regels:
      1.  Onbekende HVAC-modes worden gefilterd.
      2.  wp_actual < 0 is fysisch onmogelijk.
      3.  Negatieve delta_t bij actieve modus is fysisch onmogelijk.
      4.  DHW-opstarttransienten: supply kouder dan tank.
      5.  UFH-transienten na DHW: supply > 30 C in UFH-mode.
      6.  dhw_top < dhw_bottom: inversie wijst op sensor- of logfout.
      7.  wp_actual < 0.15 kW terwijl HVAC actief: compressor draait niet.
      8.  Deelkwartieren onder 70% van de modusspecifieke mediaan.
      9.  Annotatie full_quarter: vorige en volgende rij hebben zelfde mode.
    """
    df = df.copy()

    numeric_cols = [
        "supply_temp",
        "return_temp",
        "room_temp",
        "dhw_top",
        "dhw_bottom",
        "wp_actual",
        "hvac_mode",
        "pv_actual",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Onbekende modes
    df = df[
        df["hvac_mode"].isin(
            [
                HvacMode.OFF.value,
                HvacMode.HEATING.value,
                HvacMode.DHW.value,
            ]
        )
    ]

    # 2. Negatief elektrisch vermogen
    df = df[df["wp_actual"].isna() | (df["wp_actual"] >= 0.0)]

    # 3. Delta_t berekenen; negatieve delta_t bij actieve modus verwijderen
    df["delta_t"] = df["supply_temp"] - df["return_temp"]
    active = df["hvac_mode"].isin([HvacMode.HEATING.value, HvacMode.DHW.value])
    df = df[~(active & df["delta_t"].notna() & (df["delta_t"] < 0.0))]

    # 4. DHW-opstarttransienten
    mask_dhw = df["hvac_mode"] == HvacMode.DHW.value
    df = df[~(mask_dhw & (df["supply_temp"] < df["dhw_bottom"] + 1.0))]

    # 5. UFH-transienten na DHW-run
    mask_ufh = df["hvac_mode"] == HvacMode.HEATING.value
    df = df[~(mask_ufh & (df["supply_temp"] > 30.0))]

    # 6. Onmogelijke tank-inversie
    if "dhw_top" in df.columns and "dhw_bottom" in df.columns:
        corrupt_dhw = (
            df["dhw_top"].notna()
            & df["dhw_bottom"].notna()
            & (df["dhw_top"] < df["dhw_bottom"])
        )
        df = df[~corrupt_dhw]

    # 7. Sluipverbruik: actief maar compressor draait niet echt
    active = df["hvac_mode"].isin([HvacMode.HEATING.value, HvacMode.DHW.value])
    df = df[~(active & df["wp_actual"].notna() & (df["wp_actual"] < 0.15))]

    mask_dhw = df["hvac_mode"] == HvacMode.DHW.value
    df = df[~(mask_dhw & (df["wp_actual"] < 0.5))]

    # 9. Annotatie: volledig kwartier (vorige en volgende rij zelfde mode)
    df["full_quarter"] = (df["hvac_mode"] == df["hvac_mode"].shift(1)) & (
        df["hvac_mode"] == df["hvac_mode"].shift(-1)
    )

    return df


# =========================================================
# HP PERFORMANCE MAP
# =========================================================


class HPPerformanceMap:
    """
    Leert per modus (UFH / DHW):
      - predict_pel(mode, t_out, t_sink)  -> elektrisch vermogen [kW]
      - predict_cop(mode, t_out, t_sink)  -> COP [-]

    Beide direct uit steady-state meetdata — geen flowfactoren.
    """

    FEATURES = ["t_out", "t_sink", "supply_temp", "delta_setpoint"]

    def __init__(self, path: str):
        self.path = Path(path)
        self.is_fitted = False

        self._pel_model_ufh = None
        self._pel_model_dhw = None
        self._cop_model_ufh = None
        self._cop_model_dhw = None

        self._delta_setpoint_ufh = 2.0  # fallback
        self._delta_setpoint_dhw = 3.0

        self._pel_min_ufh = 0.4
        self._pel_max_ufh = 2.5
        self._pel_min_dhw = 0.6
        self._pel_max_dhw = 3.0

        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            d = joblib.load(self.path)
            self._pel_model_ufh = d.get("pel_ufh")
            self._pel_model_dhw = d.get("pel_dhw")
            self._cop_model_ufh = d.get("cop_ufh")
            self._cop_model_dhw = d.get("cop_dhw")
            self._setpoint_ufh = d.get("setpoint_ufh", 2.0)
            self._setpoint_dhw = d.get("setpoint_dhw", 3.0)
            self._pel_min_ufh = d.get("pel_min_ufh", 0.4)
            self._pel_max_ufh = d.get("pel_max_ufh", 2.5)
            self._pel_min_dhw = d.get("pel_min_dhw", 0.6)
            self._pel_max_dhw = d.get("pel_max_dhw", 3.0)
            self.is_fitted = True
            logger.info(
                f"[PerfMap] Geladen. "
                f"UFH [{self._pel_min_ufh:.2f}-{self._pel_max_ufh:.2f}] kW  "
                f"DHW [{self._pel_min_dhw:.2f}-{self._pel_max_dhw:.2f}] kW"
            )
        except Exception as e:
            logger.warning(f"[PerfMap] Laden mislukt: {e}")

    def _save(self):
        joblib.dump(
            {
                "pel_ufh": self._pel_model_ufh,
                "pel_dhw": self._pel_model_dhw,
                "cop_ufh": self._cop_model_ufh,
                "cop_dhw": self._cop_model_dhw,
                "setpoint_ufh": self._setpoint_ufh,
                "setpoint_dhw": self._setpoint_dhw,
                "pel_min_ufh": self._pel_min_ufh,
                "pel_max_ufh": self._pel_max_ufh,
                "pel_min_dhw": self._pel_min_dhw,
                "pel_max_dhw": self._pel_max_dhw,
            },
            self.path,
        )

    def train(self, df: pd.DataFrame):
        df = clean_thermal_data(df)

        for col in [
            "wp_actual",
            "supply_temp",
            "return_temp",
            "room_temp",
            "dhw_bottom",
            "temp",
            "delta_t",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._train_mode(df, HvacMode.HEATING.value, "room_temp", "UFH")
        self._train_mode(df, HvacMode.DHW.value, "dhw_bottom", "DHW")

        self.is_fitted = True
        self._save()
        logger.info("[PerfMap] Training compleet.")

    def _train_mode(self, df, mode_val, sink_col, label):
        base = df[df["hvac_mode"] == mode_val]
        logger.info(f"[PerfMap] {label}: {len(base)} rijen totaal in modus")
        logger.info(
            f"[PerfMap] {label}: {base['full_quarter'].sum()} full_quarter rijen"
        )
        logger.info(
            f"[PerfMap] {label}: {base['wp_actual'].notna().sum()} met wp_actual"
        )
        logger.info(f"[PerfMap] {label}: {base[sink_col].notna().sum()} met {sink_col}")
        logger.info(f"[PerfMap] {label}: {base['delta_t'].notna().sum()} met delta_t")
        logger.info(
            f"[PerfMap] {label}: {(base['target_setpoint'] > 0).sum()} met target_setpoint"
        )

        if label == "DHW":
            mask = (
                (df["hvac_mode"] == mode_val)
                & df["wp_actual"].notna()
                & (df["wp_actual"] > 0.5)  # actief vermogen
                & df["delta_t"].notna()
                & (df["delta_t"] > 2.0)  # duidelijke delta_t
                & df[sink_col].notna()
                & df["temp"].notna()
                # geen full_quarter eis
            )
        else:
            mask = (
                (df["hvac_mode"] == mode_val)
                & df["full_quarter"]
                & df["wp_actual"].notna()
                & (df["wp_actual"] > 0.15)
                & df["delta_t"].notna()
                & (df["delta_t"] > 0.5)
                & df[sink_col].notna()
                & df["temp"].notna()
            )

        d = df[mask].copy()
        d["t_out"] = d["temp"]
        d["t_sink"] = d[sink_col]
        d["supply_temp"] = d["supply_temp"]
        d["delta_setpoint"] = (d["target_setpoint"] - d["supply_temp"]).clip(-10, 45)

        n = len(d)
        if n < 10:
            logger.warning(f"[PerfMap] {label}: te weinig data, sla over.")
            return
        else:
            logger.info(f"[PerfMap] {label}: {n} steady-state rijen.")

        # Leer ook de typische setpoint per modus
        wp_setpoint_median = float(d["target_setpoint"].median())
        if label == "UFH":
            self._setpoint_ufh = float(np.clip(wp_setpoint_median, 20.0, 35.0))
        else:
            self._setpoint_dhw = float(np.clip(wp_setpoint_median, 45.0, 65.0))

        logger.info(
            f"[PerfMap] {label}: target_setpoint mediaan={wp_setpoint_median:.1f}°C"
        )
        logger.info(
            f"[PerfMap] {label}: wp_actual mediaan={d['wp_actual'].median():.3f} kW"
        )
        logger.info(f"[PerfMap] {label}: delta_t mediaan={d['delta_t'].median():.3f} K")
        logger.info(
            f"[PerfMap] {label}: supply_temp mediaan={d['supply_temp'].median():.1f} °C"
        )
        logger.info(
            f"[PerfMap] {label}: return_temp mediaan={d['return_temp'].median():.1f} °C"
        )
        logger.info(f"[PerfMap] {label}: sink mediaan={d[sink_col].median():.1f} °C")

        # Gebruik ALTIJD de fysische flow-factor voor P_th berekening
        # De flow-factor is een materiaaleigenschap van water, geen leerbare parameter
        factor_physical = FACTOR_UFH if label == "UFH" else FACTOR_DHW

        d["p_th"] = d["delta_t"] * factor_physical
        d["cop"] = (d["p_th"] / d["wp_actual"]).replace([np.inf, -np.inf], np.nan)

        logger.info(f"[PerfMap] {label}: p_th mediaan={d['p_th'].median():.3f} kW")
        logger.info(
            f"[PerfMap] {label}: cop bereik={d['cop'].quantile(0.1):.2f} - {d['cop'].quantile(0.9):.2f}"
        )

        if len(d) < 10:
            logger.warning(f"[PerfMap] {label}: te weinig data na sanity-filter.")
            return

        # Operationele grenzen leren
        pel_min = float(d["wp_actual"].quantile(0.05))
        pel_max = float(d["wp_actual"].quantile(0.95))

        if label == "UFH":
            self._pel_min_ufh = float(np.clip(pel_min, 0.3, 1.0))
            self._pel_max_ufh = float(np.clip(pel_max, 1.0, 4.0))
        else:
            self._pel_min_dhw = float(np.clip(pel_min, 0.4, 1.5))
            self._pel_max_dhw = float(np.clip(pel_max, 1.5, 5.0))

        X = d[self.FEATURES]
        y_pel = d["wp_actual"]
        y_cop = d["cop"]

        # pel_model = RandomForestRegressor(
        #     n_estimators=150, max_depth=7, min_samples_leaf=8, random_state=42
        # ).fit(X, y_pel)
        #
        # cop_model = RandomForestRegressor(
        #     n_estimators=150, max_depth=7, min_samples_leaf=8, random_state=42
        # ).fit(X, y_cop)

        pel_model, _, _ = ModelSelector.select(X, y_pel, f"{label} P_el")
        cop_model, _, _ = ModelSelector.select(X, y_cop, f"{label} COP")

        scores_pel = cross_val_score(pel_model, X, y_pel, cv=5, scoring="r2")
        scores_cop = cross_val_score(cop_model, X, y_cop, cv=5, scoring="r2")
        logger.info(
            f"[PerfMap] {label} P_el R2={scores_pel.mean():.3f}+-{scores_pel.std():.3f}  "
            f"COP R2={scores_cop.mean():.3f}+-{scores_cop.std():.3f}  "
            f"COP mediaan={y_cop.median():.2f}"
        )

        if label == "UFH":
            self._pel_model_ufh = pel_model
            self._cop_model_ufh = cop_model
        else:
            self._pel_model_dhw = pel_model
            self._cop_model_dhw = cop_model

    def predict_pel(
        self,
        mode: int,
        t_out: float,
        t_sink: float,
        supply_temp: float = None,
        delta_setpoint: float = None,
    ) -> float:

        if supply_temp is None:
            supply_temp = t_sink + 5.0
        if delta_setpoint is None:
            delta_setpoint = 2.0

        if mode == HvacMode.HEATING.value:
            model, lo, hi = self._pel_model_ufh, self._pel_min_ufh, self._pel_max_ufh
            default = float(np.clip(1.2 - 0.02 * t_out, lo, hi))
        else:
            model, lo, hi = self._pel_model_dhw, self._pel_min_dhw, self._pel_max_dhw
            default = float(np.clip(1.8 - 0.02 * t_out, lo, hi))

        if model is None:
            return default

        X = pd.DataFrame(
            [
                [
                    t_out,
                    t_sink,
                    float(supply_temp),
                    float(np.clip(delta_setpoint, -5, 20)),
                ]
            ],
            columns=self.FEATURES,
        )
        return float(np.clip(model.predict(X)[0], lo, hi))

    def predict_cop(
        self,
        mode: int,
        t_out: float,
        t_sink: float,
        supply_temp: float = None,
        delta_setpoint: float = None,
    ) -> float:

        if supply_temp is None:
            supply_temp = t_sink + 5.0
        if delta_setpoint is None:
            delta_setpoint = 2.0

        if mode == HvacMode.HEATING.value:
            model, default = self._cop_model_ufh, 3.5
        else:
            model, default = self._cop_model_dhw, 2.5

        if model is None:
            return default

        X = pd.DataFrame(
            [
                [
                    t_out,
                    t_sink,
                    float(supply_temp),
                    float(np.clip(delta_setpoint, -10, 45)),
                ]
            ],
            columns=self.FEATURES,
        )
        return float(np.clip(model.predict(X)[0], 1.2, 8.0))

    def predict_p_th(self, mode: int, t_out: float, t_sink: float) -> float:
        return self.predict_pel(mode, t_out, t_sink) * self.predict_cop(
            mode, t_out, t_sink
        )

    # Backwards-compat
    def get_pel_limits(self, t_out: float):
        return self._pel_min_ufh, self._pel_max_ufh


# =========================================================
# SYSTEM IDENTIFICATOR
# =========================================================


class SystemIdentificator:
    def __init__(self, path, tank_liters):
        self.path = Path(path)
        self.C_tank = tank_liters * 0.001163
        self.R = 15.0
        self.C = 30.0
        self.K_emit = 0.15
        self.K_tank = 0.25
        self.K_loss_dhw = 0.01
        self.ufh_lag_steps = 4
        self.C_air = 3.0  # kWh/K  — lucht + lichte oppervlakken (snelle massa)
        self.C_mass = 27.0  # kWh/K  — vloer + beton (trage massa)
        self.R_im = 1.2  # K/kW   — koppeling massa ↔ lucht
        self.R_oa = 15.0  # K/kW   — verlies lucht → buiten (= R initieel)
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            d = joblib.load(self.path)
            self.R = d.get("R", 15.0)
            self.C = d.get("C", 30.0)
            self.K_emit = d.get("K_emit", 0.15)
            self.K_tank = d.get("K_tank", 0.25)
            self.K_loss_dhw = d.get("K_loss_dhw", 0.01)
            self.ufh_lag_steps = d.get("ufh_lag_steps", 4)
            self.C_air = d.get("C_air", 3.0)
            self.C_mass = d.get("C_mass", 27.0)
            self.R_im = d.get("R_im", 1.2)
            self.R_oa = d.get("R_oa", self.R)
            logger.info(
                f"[SysID] Geladen: R={self.R:.1f} C={self.C:.1f} "
                f"K_emit={self.K_emit:.3f} K_tank={self.K_tank:.3f} "
                f"Lag={self.ufh_lag_steps * 15}m K_loss={self.K_loss_dhw:.4f} "
                f"C_air={self.C_air:.2f} C_mass={self.C_mass:.2f} "
                f"R_im={self.R_im:.3f} R_oa={self.R_oa:.2f}"
            )
        except Exception as e:
            logger.error(f"[SysID] Laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df_proc = clean_thermal_data(df).sort_values("timestamp")

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

        df_proc["delta_t"] = (df_proc["supply_temp"] - df_proc["return_temp"]).clip(
            lower=0.0
        )
        df_proc["wp_output"] = np.select(
            [
                df_proc["hvac_mode"] == HvacMode.HEATING.value,
                df_proc["hvac_mode"] == HvacMode.DHW.value,
            ],
            [df_proc["delta_t"] * FACTOR_UFH, df_proc["delta_t"] * FACTOR_DHW],
            default=0.0,
        )

        df_1h = (
            df_proc.set_index("timestamp")
            .resample("1h")
            .mean()
            .dropna(subset=["room_temp", "temp", "pv_actual", "wp_output"])
            .reset_index()
        )
        df_1h["dT_next"] = df_1h["room_temp"].shift(-1) - df_1h["room_temp"]
        df_1h["delta_T_env"] = df_1h["room_temp"] - df_1h["temp"]

        self._fit_rc(df_1h)
        self._fit_floor_lag(df_proc)
        self._fit_k_emit(df_proc)
        self._fit_k_tank(df_proc)
        self._fit_dhw_loss(df_proc)
        self._fit_two_state()

        logger.info(
            f"[SysID] Klaar: R={self.R:.1f} C={self.C:.1f} "
            f"K_emit={self.K_emit:.3f} K_tank={self.K_tank:.3f} "
            f"K_loss={self.K_loss_dhw:.4f}"
        )
        joblib.dump(
            {
                "R": self.R,
                "C": self.C,
                "K_emit": self.K_emit,
                "K_tank": self.K_tank,
                "K_loss_dhw": self.K_loss_dhw,
                "ufh_lag_steps": self.ufh_lag_steps,
                "C_air": self.C_air,
                "C_mass": self.C_mass,
                "R_im": self.R_im,
                "R_oa": self.R_oa,
            },
            self.path,
        )

    def _fit_rc(self, df_1h: pd.DataFrame):
        tau_inv = None

        mask_cool = (
            (df_1h["wp_output"] < 0.1)
            & (df_1h["pv_actual"] < 0.05)
            & (df_1h["dT_next"] < -0.01)
            & (df_1h["delta_T_env"] > 3.0)
        )
        df_cool = df_1h[mask_cool]

        if len(df_cool) > 10:
            lr = LinearRegression(fit_intercept=False).fit(
                -df_cool[["delta_T_env"]], df_cool["dT_next"]
            )
            if lr.coef_[0] > 0:
                tau = 1.0 / lr.coef_[0]
                if 20 < tau < 1000:
                    tau_inv = lr.coef_[0]
                    logger.info(
                        f"[SysID] Tau(RC)={tau:.1f} uur ({len(df_cool)} punten)"
                    )

        if tau_inv is not None:
            mask_heat = (df_1h["wp_output"] > 0.5) & (df_1h["dT_next"] > 0.01)
            df_heat = df_1h[mask_heat]
            if len(df_heat) > 10:
                y_adj = df_heat["dT_next"] + tau_inv * df_heat["delta_T_env"]
                lr2 = LinearRegression(fit_intercept=False).fit(
                    df_heat[["wp_output"]], y_adj
                )
                if lr2.coef_[0] > 0:
                    new_C = 1.0 / lr2.coef_[0]
                    new_R = (1.0 / tau_inv) / new_C
                    if 10.0 < new_C < 200.0 and 2.0 < new_R < 60.0:
                        self.C = float(new_C)
                        self.R = float(new_R)
                        logger.info(
                            f"[SysID] R={self.R:.2f} C={self.C:.2f} (gesplitst)"
                        )
                        return
            tau_inv = None

        # Multivariate fallback
        df_m = df_1h.dropna(subset=["dT_next", "wp_output", "delta_T_env"])
        df_m = df_m[(df_m["wp_output"] > 0.1) | (np.abs(df_m["delta_T_env"]) > 3.0)]
        if len(df_m) > 20:
            lr = LinearRegression(fit_intercept=False).fit(
                df_m[["wp_output", "delta_T_env"]], df_m["dT_next"]
            )
            b1, b2 = lr.coef_[0], -lr.coef_[1]
            if b1 > 0.001 and b2 > 0.001:
                self.C = float(np.clip(1.0 / b1, 10.0, 200.0))
                self.R = float(np.clip(b1 / b2, 2.0, 60.0))
                logger.info(f"[SysID] R={self.R:.2f} C={self.C:.2f} (fallback)")
            else:
                logger.error("[SysID] R/C training mislukt; defaults behouden.")
        else:
            logger.error("[SysID] Te weinig data voor R/C; defaults behouden.")

    def _fit_floor_lag(self, df_proc: pd.DataFrame):
        df_lag = df_proc[
            ["timestamp", "room_temp", "wp_output", "full_quarter", "hvac_mode"]
        ].copy()
        df_lag.loc[
            (df_lag["hvac_mode"] == HvacMode.HEATING.value) & (~df_lag["full_quarter"]),
            "wp_output",
        ] = np.nan

        df_15 = (
            df_lag[["timestamp", "room_temp", "wp_output"]]
            .set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna(subset=["room_temp"])
            .reset_index()
        )
        df_15["room_smooth"] = df_15["room_temp"].rolling(4, center=True).mean()
        df_15["dT_trend"] = df_15["room_smooth"].diff(4)

        best_lag, best_corr = 0, -1.0
        for lag in range(0, 17):
            col = f"wp_lag_{lag}"
            df_15[col] = df_15["wp_output"].shift(lag)
            corr = df_15["dT_trend"].corr(df_15[col])
            if pd.notna(corr) and corr > best_corr:
                best_corr, best_lag = corr, lag

        if best_corr < 0.1:
            logger.warning(
                f"[SysID] Geen duidelijke vloerlag (corr={best_corr:.2f}); fallback 1 uur."
            )
            self.ufh_lag_steps = 4
        else:
            self.ufh_lag_steps = int(np.clip(best_lag, 2, 16))
            logger.info(
                f"[SysID] Vloerlag={self.ufh_lag_steps * 15} min (corr={best_corr:.2f})"
            )

    def _fit_k_emit(self, df_proc: pd.DataFrame):
        mask = (
            (df_proc["hvac_mode"] == HvacMode.HEATING.value)
            & df_proc["full_quarter"]
            & (df_proc["wp_output"] > 0.5)
            & (df_proc["supply_temp"] < 30)
            & (df_proc["supply_temp"] > df_proc["room_temp"] + 1)
        )
        d = df_proc[mask].copy()
        if len(d) > 20:
            t_avg = (d["supply_temp"] + d["return_temp"]) / 2
            delta = t_avg - d["room_temp"]
            valid = delta > 0.5
            if valid.any():
                k = d.loc[valid, "wp_output"] / delta[valid]
                self.K_emit = float(np.clip(k.median(), 0.05, 1.5))
                logger.info(f"[SysID] K_emit={self.K_emit:.3f}")

    def _fit_k_tank(self, df_proc: pd.DataFrame):
        mask = (
            (df_proc["hvac_mode"] == HvacMode.DHW.value)
            & df_proc["full_quarter"]
            & (df_proc["wp_output"] > 0.8)
            & (df_proc["supply_temp"] > df_proc["dhw_bottom"] + 1)
            & df_proc["return_temp"].notna()
        )
        d = df_proc[mask].copy()
        if len(d) > 10:
            t_avg = (d["supply_temp"] + d["return_temp"]) / 2
            delta = t_avg - d["dhw_bottom"]
            valid = delta > 2.0
            if valid.any():
                k = d.loc[valid, "wp_output"] / delta[valid]
                self.K_tank = float(np.clip(k.median(), 0.15, 2.0))
                logger.info(f"[SysID] K_tank={self.K_tank:.3f}")

    def _fit_dhw_loss(self, df_proc: pd.DataFrame):
        df_loss = df_proc.sort_values("timestamp").copy()
        df_loss["t_tank"] = (df_loss["dhw_top"] + df_loss["dhw_bottom"]) / 2.0
        df_loss["dt_hours"] = df_loss["timestamp"].diff().dt.total_seconds() / 3600.0
        df_loss["dT_tank"] = df_loss["t_tank"].diff()

        mask = (
            (df_loss["hvac_mode"] != HvacMode.DHW.value)
            & (df_loss["wp_output"] < 0.1)
            & (df_loss["dT_tank"] < 0)
            & (df_loss["dt_hours"] > 0.1)
            & (df_loss["dT_tank"] > -0.3)
            & (df_loss["dt_hours"] < 1.0)
        )
        d = df_loss[mask].copy()
        if len(d) > 20:
            t_diff = (d["t_tank"] - d["room_temp"]).clip(lower=1.0)
            loss = -(d["dT_tank"] / d["dt_hours"]) / t_diff
            self.K_loss_dhw = float(np.clip(loss.quantile(0.25), 0.001, 0.05))
            logger.info(f"[SysID] K_loss_dhw={self.K_loss_dhw:.4f}")

    def _fit_two_state(self):
        """
        2-state parameterschatting.

        Wat zeker is (uit data):
          R_oa = R            — verliespad lucht → buiten
          C_air + C_mass = C  — massa-behoud
          τ_lag = R_im × C_air — gemeten vertraging

        Wat een gefundeerde prior is (niet meetbaar zonder vloersensor):
          C_air ≈ 8% van C    — lucht + lichte oppervlakken, zware bouw
          R_im afgeleid uit τ_lag en C_air

        De solar-event methode wordt NIET gebruikt: de meting (0.29 kWh/K)
        valt consequent buiten het fysisch verdedigbare bereik, wat betekent
        dat de proxy (PV × 0.15) niet representatief is voor dit huis.
        """

        # ── 1. R_oa: zeker, identiek aan totale verliesweerstand ─────────────
        self.R_oa = self.R

        # ── 2. C_air: prior op basis van bouwtype ────────────────────────────
        # Zware bouw (beton/baksteen + UFH): lucht + meubels + lichte laag
        # is typisch 5-12% van totale thermische massa.
        # We gebruiken 8% als centraal anker voor zware bouw.
        # Zonder vloersensor is dit de meest eerlijke schatting.
        C_air_prior_frac = 0.08
        C_air_prior = C_air_prior_frac * self.C

        # Harde fysische grenzen:
        #   Ondergrens: lucht alleen in ~100m² woning ≈ 0.35 kWh/K, met meubels ~0.6
        #   Bovengrens: 15% van C (lichte bouw maximum)
        c_air_lower = max(0.6, 0.04 * self.C)
        c_air_upper = 0.15 * self.C
        self.C_air = float(np.clip(C_air_prior, c_air_lower, c_air_upper))

        # ── 3. R_im: afgeleid uit gemeten lag en C_air ────────────────────────
        # τ_lag = R_im × C_air  →  R_im = τ_lag / C_air
        # Dit is de enige vergelijking die R_im direct koppelt aan meetdata.
        tau_lag_h = (self.ufh_lag_steps * 15) / 60.0  # uren
        R_im_from_lag = tau_lag_h / max(self.C_air, 0.5)

        # Sanity check: R_im moet kleiner zijn dan 1/K_emit (totale weerstand)
        R_total = 1.0 / max(self.K_emit, 0.05)
        R_im_max = 0.85 * R_total  # vloer→lucht is hooguit 85% van totaal

        self.R_im = float(np.clip(R_im_from_lag, 0.3, R_im_max))

        # ── 4. C_mass: rest ───────────────────────────────────────────────────
        self.C_mass = float(max(self.C - self.C_air, 5.0))

        # ── 5. Consistentiecheck ─────────────────────────────────────────────
        tau_air_implied = self.R_im * self.C_air * 60  # minuten
        tau_mass_implied = self.R_im * self.C_mass * 60

        logger.info(
            f"[SysID] 2-state: "
            f"C_air={self.C_air:.2f} kWh/K ({self.C_air / self.C * 100:.1f}%), "
            f"C_mass={self.C_mass:.2f} kWh/K, "
            f"R_im={self.R_im:.3f} K/kW, R_oa={self.R_oa:.2f} K/kW | "
            f"τ_lucht={tau_air_implied:.0f}min (input lag={self.ufh_lag_steps * 15}min), "
            f"τ_massa={tau_mass_implied:.0f}min"
        )

        # Waarschuw als R_im geclipt werd (prior en lag zijn inconsistent)
        if R_im_from_lag > R_im_max:
            logger.warning(
                f"[SysID] R_im uit lag ({R_im_from_lag:.3f}) > R_total ({R_total:.3f}). "
                f"Lag-meting en K_emit zijn inconsistent. R_im geclipped naar {self.R_im:.3f}. "
                f"Overweeg C_air_prior_frac te verhogen."
            )


# =========================================================
# HYDRAULIC PREDICTOR
# =========================================================


class HydraulicPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.is_fitted = False

        self.model_supply_ufh = None
        self.model_supply_dhw = None
        self.learned_lift_ufh = 0.5
        self.learned_lift_dhw = 3.0
        self.learned_ufh_slope = 0.4
        self.dhw_delta_slope = 0.0
        self.dhw_delta_base = 5.0

        self.features = ["wp_output", "temp", "sink_temp"]
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            d = joblib.load(self.path)
            self.model_supply_ufh = d.get("supply_ufh")
            self.model_supply_dhw = d.get("supply_dhw")
            self.learned_lift_ufh = d.get("lift_ufh", 0.5)
            self.learned_lift_dhw = d.get("lift_dhw", 3.0)
            self.learned_ufh_slope = d.get("ufh_slope", 0.4)
            self.dhw_delta_slope = d.get("dhw_slope", 0.0)
            self.dhw_delta_base = d.get("dhw_base", 5.0)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"[Hydraulic] Laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = clean_thermal_data(df)
        df = df.dropna(subset=["delta_t", "supply_temp", "return_temp"])
        df = df[(df["delta_t"] > 0.1) & (df["delta_t"] < 15.0)].copy()

        conditions = [
            df["hvac_mode"] == HvacMode.HEATING.value,
            df["hvac_mode"] == HvacMode.DHW.value,
        ]
        df["sink_temp"] = np.select(
            conditions, [df["room_temp"], df["dhw_bottom"]], default=20.0
        )
        df["p_th_raw"] = np.select(
            conditions,
            [df["delta_t"] * FACTOR_UFH, df["delta_t"] * FACTOR_DHW],
            default=0.0,
        )
        df["wp_output"] = df["p_th_raw"]

        # UFH
        mask_ufh = (
            (df["hvac_mode"] == HvacMode.HEATING.value)
            & (df["p_th_raw"] > 0.5)
            & (df["supply_temp"] > df["room_temp"] + 1.0)
        )
        df_ufh = df[mask_ufh].dropna(
            subset=self.features + ["supply_temp", "return_temp"]
        )

        if len(df_ufh) > 15:
            # ← factor_ufh NIET meer leren, altijd fysische constante
            self.learned_lift_ufh = max(
                0.2, (df_ufh["return_temp"] - df_ufh["room_temp"]).quantile(0.05)
            )

            mask_curve = (
                mask_ufh
                & (df["temp"] < 15.0)
                & (df["supply_temp"] > df["room_temp"] + 2.0)
            )
            df_curve = df[mask_curve].copy()
            if len(df_curve) > 50:
                slopes = (
                    df_curve["supply_temp"]
                    - df_curve["room_temp"]
                    - self.learned_lift_ufh
                ) / (20.0 - df_curve["temp"]).clip(lower=0.1)
                self.learned_ufh_slope = float(np.clip(slopes.median(), 0.1, 1.5))

            self.model_supply_ufh = RandomForestRegressor(
                n_estimators=50, max_depth=6
            ).fit(df_ufh[self.features], df_ufh["supply_temp"])

        # DHW
        mask_dhw = (
            (df["hvac_mode"] == HvacMode.DHW.value)
            & df["full_quarter"]
            & (df["p_th_raw"] > 1.5)
            & (df["delta_t"] > 2.0)
            & (df["supply_temp"] > df["dhw_bottom"] + 3.0)
        )
        df_dhw = (
            df[mask_dhw]
            .dropna(subset=self.features + ["supply_temp", "return_temp"])
            .copy()
        )

        if len(df_dhw) > 10:
            # ← factor_dhw NIET meer leren, altijd fysische constante
            self.learned_lift_dhw = max(
                1.0, (df_dhw["return_temp"] - df_dhw["dhw_bottom"]).quantile(0.10)
            )

            df_dhw["t_bin"] = df_dhw["temp"].round()
            tops = [
                (g["p_th_raw"].quantile(0.90), t)
                for t, g in df_dhw.groupby("t_bin")
                if len(g) >= 3
            ]

            if len(tops) >= 3:
                df_tops = pd.DataFrame(tops, columns=["p_th_raw", "temp"])
                reg = LinearRegression().fit(df_tops[["temp"]], df_tops["p_th_raw"])
                p_slope = float(np.clip(reg.coef_[0], -0.15, 0.25))
                p_base = float(reg.intercept_)
            else:
                p_slope = 0.0
                p_base = (
                    df_dhw["delta_t"].median() * FACTOR_DHW
                )  # ← ook hier vaste constante

            self.dhw_delta_slope = p_slope / FACTOR_DHW  # ← vaste constante
            self.dhw_delta_base = p_base / FACTOR_DHW  # ← vaste constante

            min_at_35 = self.dhw_delta_base + self.dhw_delta_slope * 35.0
            if min_at_35 < 1.5:
                self.dhw_delta_base += 1.5 - min_at_35

            self.model_supply_dhw = RandomForestRegressor(
                n_estimators=50, max_depth=6
            ).fit(df_dhw[self.features], df_dhw["supply_temp"])

        self.is_fitted = True
        joblib.dump(
            {
                "supply_ufh": self.model_supply_ufh,
                "supply_dhw": self.model_supply_dhw,
                "lift_ufh": self.learned_lift_ufh,
                "lift_dhw": self.learned_lift_dhw,
                "ufh_slope": self.learned_ufh_slope,
                "dhw_slope": self.dhw_delta_slope,
                "dhw_base": self.dhw_delta_base,
            },
            self.path,
        )
        logger.info("[Hydraulic] Training compleet.")

    def predict_supply(self, mode, p_th, t_out, t_sink):
        if mode == "UFH":
            factor, min_lift, max_safe = FACTOR_UFH, self.learned_lift_ufh, 30.0
        else:
            factor, min_lift, max_safe = FACTOR_DHW, self.learned_lift_dhw, 60.0

        delta_t_calc = p_th / factor if p_th > 0 else 0.0
        min_hard = t_sink + min_lift + delta_t_calc
        prediction = min_hard

        if self.is_fitted:
            data = pd.DataFrame([[p_th, t_out, t_sink]], columns=self.features)
            try:
                ml_model = (
                    self.model_supply_ufh if mode == "UFH" else self.model_supply_dhw
                )
                if ml_model is not None:
                    prediction = float(ml_model.predict(data)[0])
            except Exception:
                pass

        return float(np.clip(prediction, min_hard, max_safe))

    def predict_delta(self, mode, p_th):
        factor = FACTOR_UFH if mode == "UFH" else FACTOR_DHW
        return p_th / factor if p_th > 0 else 0.0

    def get_ufh_slope(self, t_out):
        return max(0.0, 20.0 - t_out) * self.learned_ufh_slope


# =========================================================
# UFH RESIDUAL PREDICTOR
# =========================================================


class UfhResidualPredictor:
    def __init__(self, path, R, C):
        self.path = Path(path)
        self.R = R
        self.C = C
        self.model = None
        self.is_fitted = False
        self.features = [
            "temp",
            "solar",
            "effective_solar",
            "shutter_room",
            "wind",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            self.model = joblib.load(self.path)
            self.is_fitted = True
            logger.info("[UFH Residual] Geladen.")
        except Exception as e:
            logger.warning(f"[UFH Residual] Laden mislukt: {e}")

    def train(self, df):
        df_proc = clean_thermal_data(df).sort_values("timestamp")

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
        mask = (df_proc["hvac_mode"] == HvacMode.HEATING.value) | (
            df_proc["wp_output"] < 0.1
        )
        df_feat = df_proc[mask].copy()
        dt = dt_hours[mask]

        t_curr = df_feat["room_temp"]
        t_next = df_feat["room_temp"].shift(-1)
        p_heat = df_feat["wp_output"]
        t_out = df_feat["temp"]
        t_model_next = t_curr + ((p_heat - (t_curr - t_out) / self.R) * dt / self.C)
        target = (t_next - t_model_next) / dt

        df_feat = add_cyclic_time_features(df_feat, "timestamp")
        df_feat["solar"] = df_feat["pv_actual"]
        df_feat["target"] = target
        df_feat["effective_solar"] = df_feat["solar"] * (
            df_feat.get("shutter_room", 100) / 100.0
        )

        train_set = df_feat[self.features + ["target"]].dropna()
        train_set = train_set[train_set["target"].between(-1.2, 1.2)]

        if len(train_set) > 10:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=15
            ).fit(train_set[self.features], train_set["target"])
            self.is_fitted = True
            joblib.dump(self.model, self.path)
            logger.info("[UFH Residual] Getraind.")

    def predict(self, forecast_df, shutters):
        if self.model is None or not self.is_fitted:
            return np.zeros(len(forecast_df))

        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        df["solar"] = df.get("power_corrected", df.get("pv_estimate", 0.0))
        df["shutter_room"] = shutters
        df["effective_solar"] = df["solar"] * (df["shutter_room"] / 100.0)

        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0

        return self.model.predict(df[self.features])


# =========================================================
# DHW RESIDUAL PREDICTOR
# =========================================================


class DhwResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.is_fitted = False
        self.features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            self.model = joblib.load(self.path)
            self.is_fitted = True
            logger.info("[DHW Residual] Geladen.")
        except Exception as e:
            logger.warning(f"[DHW Residual] Laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy().sort_values("timestamp")
        df["dhw_diff"] = df["dhw_top"].diff()
        mask_shower = (df["hvac_mode"] != HvacMode.DHW.value) & (df["dhw_diff"] < -0.5)
        df["demand"] = 0.0
        df.loc[mask_shower, "demand"] = df["dhw_diff"].abs()
        df = add_cyclic_time_features(df, "timestamp")

        if len(df) > 100:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42
            ).fit(df[self.features], df["demand"])
            self.is_fitted = True
            joblib.dump(self.model, self.path)
            logger.info("[DHW Residual] Getraind.")
        else:
            logger.warning("[DHW Residual] Te weinig data.")

    def predict(self, forecast_df):
        if self.model is None or not self.is_fitted:
            return np.zeros(len(forecast_df))

        df_feat = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        predictions = self.model.predict(df_feat[self.features])
        return np.where(predictions < 0.8, 0.0, predictions)


# =========================================================
# PHYSICS LINEARIZER  (SLP-hulpklasse voor ThermalMPC)
# =========================================================


class PhysicsLinearizer:
    """
    Berekent per tijdstap de bevroren P_el en COP voor de MILP.
    Convergeert via een verschilcheck op P_el (tol kW).
    """

    def __init__(
        self,
        perf_map: HPPerformanceMap,
        hydraulic: HydraulicPredictor,
        horizon: int = 96,
        max_iter: int = 6,
        tol: float = 0.05,
    ):
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.horizon = horizon
        self.max_iter = max_iter
        self.tol = tol

    def compute(
        self,
        t_out_arr: np.ndarray,
        t_room_arr: np.ndarray,
        t_dhw_arr: np.ndarray,
    ) -> tuple:
        T = self.horizon
        p_el_ufh = np.zeros(T)
        cop_ufh = np.zeros(T)
        p_el_dhw = np.zeros(T)
        cop_dhw = np.zeros(T)
        sup_ufh_arr = np.zeros(T)
        sup_dhw_arr = np.zeros(T)

        for t in range(T):
            t_out = float(t_out_arr[t])
            t_room = float(t_room_arr[t])
            t_dhw = float(t_dhw_arr[t])

            # Supply projecteren via hydraulic curve
            p_th_ufh_est = self.perf_map.predict_p_th(
                HvacMode.HEATING.value, t_out, t_room
            )
            p_th_dhw_est = self.perf_map.predict_p_th(HvacMode.DHW.value, t_out, t_dhw)

            sup_ufh = self.hydraulic.predict_supply("UFH", p_th_ufh_est, t_out, t_room)
            sup_dhw = self.hydraulic.predict_supply("DHW", p_th_dhw_est, t_out, t_dhw)

            sup_ufh_arr[t] = sup_ufh
            sup_dhw_arr[t] = sup_dhw

            # wp_setpoint is vast (bijv. 55°C), supply loopt op met t_dhw
            # delta_setpoint = wp_setpoint - supply_temp
            # supply_temp ≈ t_dhw + hydraulische lift (loopt op met t_dhw)
            # delta neemt automatisch af naarmate t_dhw stijgt
            delta_dhw = float(np.clip(self.perf_map._setpoint_dhw - sup_dhw, 0.5, 25.0))

            # Zelfde voor UFH (minder dramatisch want t_room varieert weinig):
            delta_ufh = float(np.clip(self.perf_map._setpoint_ufh - sup_ufh, 0.5, 8.0))

            p_el_ufh[t] = self.perf_map.predict_pel(
                HvacMode.HEATING.value, t_out, t_room, sup_ufh, delta_ufh
            )
            cop_ufh[t] = self.perf_map.predict_cop(
                HvacMode.HEATING.value, t_out, t_room, sup_ufh, delta_ufh
            )
            p_el_dhw[t] = self.perf_map.predict_pel(
                HvacMode.DHW.value, t_out, t_dhw, sup_dhw, delta_dhw
            )
            cop_dhw[t] = self.perf_map.predict_cop(
                HvacMode.DHW.value, t_out, t_dhw, sup_dhw, delta_dhw
            )

        return p_el_ufh, cop_ufh, p_el_dhw, cop_dhw, sup_ufh_arr, sup_dhw_arr

    def has_converged(
        self,
        p_el_ufh_prev,
        p_el_dhw_prev,
        p_el_ufh_new,
        p_el_dhw_new,
        ufh_on=None,
        dhw_on=None,
    ) -> bool:

        if ufh_on is not None and np.any(ufh_on > 0.5):
            delta_ufh = float(
                np.max(np.abs((p_el_ufh_new - p_el_ufh_prev)[ufh_on > 0.5]))
            )
        else:
            delta_ufh = float(np.max(np.abs(p_el_ufh_new - p_el_ufh_prev)))

        if dhw_on is not None and np.any(dhw_on > 0.5):
            delta_dhw = float(
                np.max(np.abs((p_el_dhw_new - p_el_dhw_prev)[dhw_on > 0.5]))
            )
        else:
            delta_dhw = float(np.max(np.abs(p_el_dhw_new - p_el_dhw_prev)))

        tol_ufh = self.tol  # 0.05 kW
        tol_dhw = self.tol  # * 3.0  # 0.15 kW — DHW mag ruimer

        converged = (delta_ufh < tol_ufh) and (delta_dhw < tol_dhw)
        logger.info(
            f"[SLP] delta UFH={delta_ufh:.4f}/{tol_ufh}  "
            f"DHW={delta_dhw:.4f}/{tol_dhw}  converged={converged}"
        )
        return converged


# =========================================================
# COMFORT COST CALCULATOR
# =========================================================


class ComfortCostCalculator:
    """
    Comfortboetes in euro per graad afwijking — dimensioneel correct.
    De boete wordt gebaseerd op de maximale elektriciteitsprijs. Hierdoor
    is het oplossen van een temperatuurtekort op een gemiddeld/goedkoop moment
    altijd goedkoper dan de comfortboete accepteren.
    """

    def __init__(
        self,
        C_room: float,
        C_tank: float,
        avg_cop_ufh: float,
        avg_cop_dhw: float,
    ):
        self.C_room = C_room
        self.C_tank = C_tank
        self.avg_cop_ufh = avg_cop_ufh
        self.avg_cop_dhw = avg_cop_dhw

    def compute(self, max_price: float) -> dict:
        # Thermodynamische kosten om de volledige thermische massa 1 graad te verwarmen
        cost_to_heat_1k_room = (self.C_room / self.avg_cop_ufh) * max_price
        cost_to_heat_1k_tank = (self.C_tank / self.avg_cop_dhw) * max_price

        # Wiskundige harde grens: Boete moet strict groter zijn dan de fysieke opwarmkosten
        # Anders kiest de optimizer voor temperatuurverlies. Epsilon = 1% extra.
        room_under = cost_to_heat_1k_room * 1.01
        tank_under = cost_to_heat_1k_tank * 1.01

        # Oververhitting boete: zonne-overschot opslaan is prima, maar niet oneindig
        # Maak dit goedkoper dan onderkoeling, zodat pre-heating toegestaan is.
        room_over = cost_to_heat_1k_room * 0.1
        tank_over = cost_to_heat_1k_tank * 0.1

        # Terminal Value = Waarde van de restwarmte op t=Horizon.
        # Exact gelijk aan de opwarmkosten, anders laat hij het huis afkoelen in de laatste uren.
        terminal_room = cost_to_heat_1k_room
        terminal_tank = cost_to_heat_1k_tank

        return {
            "room_under": room_under,
            "room_over": room_over,
            "tank_under": tank_under,
            "tank_over": tank_over,
            "terminal_room": terminal_room,
            "terminal_tank": terminal_tank,
        }
