import logging
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import curve_fit
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
    Verwijdert fysiek corrupte rijen en labelt kwalitatieve data voor training.

    Strategie:
    - Verwijder (drop): Sensorfouten, onmogelijke inversies, puur sluipverbruik.
    - Label (vlag): Mix-metingen en transienten (niet verwijderen t.b.v. RC-tijdlijn).
    """
    df = df.copy()

    # 0. Kolomnamen standaardiseren (match met Database)
    numeric_cols = [
        "supply_temp",
        "return_temp",
        "room_temp",
        "dhw_top",
        "dhw_bottom",
        "wp_actual",
        "hvac_mode",
        "pv_actual",
        "grid_import",
        "grid_export",
        "wp_ufh",
        "wp_dhw",
        "wp_leg",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Geldige modi
    df = df[
        df["hvac_mode"].isin(
            [
                HvacMode.OFF.value,
                HvacMode.HEATING.value,
                HvacMode.DHW.value,
                HvacMode.COOLING.value,
                HvacMode.LEGIONELLA_PREVENTION.value,
            ]
        )
    ]

    # 3. Verwijder ruis (pompjes/standby) PER KOLOM
    # Alles onder 150W is geen compressorwerk. Door dit op 0 te zetten,
    # worden de ratio-berekeningen hieronder veel zuiverder.
    for col in ["wp_ufh", "wp_dhw", "wp_leg"]:
        if col in df.columns:
            df.loc[df[col] < 0.15, col] = 0.0

    # 2. Herbereken wp_actual voor 100% consistentie
    df["wp_actual"] = (
        df["wp_ufh"].fillna(0) + df["wp_dhw"].fillna(0) + df["wp_leg"].fillna(0)
    ).infer_objects(copy=False)

    # 4. Dominantie en Purity (De "Mix" check)
    # We kijken welk aandeel de stroom heeft t.o.v. het totaal.
    # Dit lost het probleem op van de "7 min DHW / 8 min UFH" kwartieren.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ufh_ratio"] = df["wp_ufh"] / df["wp_actual"]
        df["dhw_ratio"] = df["wp_dhw"] / df["wp_actual"]

    # Een meting is 'Pure' als >90% van de stroom naar één doel ging.
    df["pure_ufh"] = (df["ufh_ratio"] > 0.9) & (df["wp_ufh"] > 0.4)
    df["pure_dhw"] = (df["dhw_ratio"] > 0.9) & (df["wp_dhw"] > 0.6)

    # 5. VERWIJDEREN: Harde sensor- en systeemfouten
    # A. Negatieve Delta T terwijl de compressor draait (>300W)
    df["delta_t"] = df["supply_temp"] - df["return_temp"]
    df = df[~((df["wp_actual"] > 0.15) & (df["delta_t"] < 0.0))]

    # B. Onmogelijke tank-inversie (Top kouder dan bodem)
    if "dhw_top" in df.columns and "dhw_bottom" in df.columns:
        corrupt_dhw = (
            df["dhw_top"].notna()
            & df["dhw_bottom"].notna()
            & (df["dhw_top"] < df["dhw_bottom"])
        )
        df = df[~corrupt_dhw]

    # C. Sluipverbruik / Standby filter
    # Als de modus actief is, maar het TOTAAL verbruik is < 150W,
    # dan draait alleen een circulatiepompje. Onbruikbaar voor elke training.
    standby = (df["hvac_mode"] != HvacMode.OFF.value) & (df["wp_actual"] < 0.15)
    df.loc[standby, ["wp_actual", "wp_ufh", "wp_dhw", "wp_leg"]] = 0.0

    # 6. LABELEN: Transienten en Steady State
    # Deze rijen worden NIET verwijderd (belangrijk voor RC-tijdlijn),
    # maar we labelen ze zodat de Performance Map ze kan negeren.

    # DHW opstart: supply is nog kouder dan de tank
    mask_dhw_start = (df["hvac_mode"] == HvacMode.DHW.value) & (
        df["supply_temp"] <= df["dhw_bottom"]
    )

    # UFH na DHW: supply is nog veel te heet van de boiler-run
    mask_ufh_after_dhw = (df["hvac_mode"] == HvacMode.HEATING.value) & (
        df["supply_temp"] > 30.0
    )

    # De heilige graal voor de Performance Map:
    df["steady_state"] = (
        (df["pure_ufh"] | df["pure_dhw"]) & ~mask_dhw_start & ~mask_ufh_after_dhw
    )

    # 7. Annotatie: volledig kwartier (voor hydraulische traagheid)
    df["full_quarter"] = (df["hvac_mode"] == df["hvac_mode"].shift(1)) & (
        df["hvac_mode"] == df["hvac_mode"].shift(-1)
    )

    # 8. Base load (voor RC-model/sluipverbruik in huis)
    df["base_load"] = (
        df["grid_import"].fillna(0)
        - df["grid_export"].fillna(0)
        + df["pv_actual"].fillna(0)
        - df["wp_actual"].fillna(0)
    ).clip(lower=0)

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

        self._setpoint_ufh = 2.0  # fallback
        self._setpoint_dhw = 3.0

        self._pel_min_ufh = 0.4
        self._pel_max_ufh = 2.5
        self._pel_min_dhw = 0.6
        self._pel_max_dhw = 3.0

        self._carnot_ufh = None  # tuple: (eta, dT_lift)
        self._carnot_dhw = None

        self._pel_ufh = None  # tuple: (a, b, c) voor p_el = a + b*t_sink + c*t_out
        self._pel_dhw = None

        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            d = joblib.load(self.path)
            self._carnot_ufh = d.get("carnot_ufh")
            self._carnot_dhw = d.get("carnot_dhw")
            self._pel_ufh = d.get("pel_ufh")
            self._pel_dhw = d.get("pel_dhw")
            self._setpoint_ufh = d.get("setpoint_ufh", 25.0)
            self._setpoint_dhw = d.get("setpoint_dhw", 55.0)
            self._pel_min_ufh = d.get("pel_min_ufh", 0.4)
            self._pel_max_ufh = d.get("pel_max_ufh", 2.5)
            self._pel_min_dhw = d.get("pel_min_dhw", 0.6)
            self._pel_max_dhw = d.get("pel_max_dhw", 3.0)
            self.is_fitted = True
            logger.info(
                f"[PerfMap] Geladen. "
                f"UFH η={self._carnot_ufh[0]:.3f} lift={self._carnot_ufh[1]:.1f}K  "
                f"DHW η={self._carnot_dhw[0]:.3f} lift={self._carnot_dhw[1]:.1f}K"
                if self._carnot_ufh and self._carnot_dhw
                else "[PerfMap] Geladen (defaults)."
            )
        except Exception as e:
            logger.warning(f"[PerfMap] Laden mislukt: {e}")

    def _save(self):
        joblib.dump(
            {
                "carnot_ufh": self._carnot_ufh,
                "carnot_dhw": self._carnot_dhw,
                "pel_ufh": self._pel_ufh,
                "pel_dhw": self._pel_dhw,
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
            "wp_ufh",
            "wp_dhw",
            "supply_temp",
            "return_temp",
            "room_temp",
            "dhw_bottom",
            "temp",
            "delta_t",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._train_mode(df, HvacMode.HEATING.value, "room_temp", "UFH", "wp_ufh")
        self._train_mode(df, HvacMode.DHW.value, "dhw_bottom", "DHW", "wp_dhw")

        self.is_fitted = True
        self._save()
        logger.info("[PerfMap] Training compleet.")

    def _train_mode(self, df, mode_val, sink_col, label, p_el_col):
        base = df[df["hvac_mode"] == mode_val].copy()
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

        mask = (
            df["steady_state"]
            & df[p_el_col].notna()
            & (df[p_el_col] > 0.15)
            & df["delta_t"].notna()
            & (df["delta_t"] > 1.0)
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
        if mode_val == HvacMode.HEATING.value:
            self._setpoint_ufh = float(np.clip(wp_setpoint_median, 20.0, 35.0))
        else:
            self._setpoint_dhw = float(np.clip(wp_setpoint_median, 45.0, 65.0))

        logger.info(
            f"[PerfMap] {label}: target_setpoint mediaan={wp_setpoint_median:.1f}°C"
        )
        logger.info(
            f"[PerfMap] {label}: wp_actual mediaan={d[p_el_col].median():.3f} kW"
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
        d["cop"] = (d["p_th"] / d[p_el_col]).replace([np.inf, -np.inf], np.nan)

        logger.info(f"[PerfMap] {label}: p_th mediaan={d['p_th'].median():.3f} kW")
        logger.info(
            f"[PerfMap] {label}: cop bereik={d['cop'].quantile(0.1):.2f} - {d['cop'].quantile(0.9):.2f}"
        )

        if len(d) < 10:
            logger.warning(f"[PerfMap] {label}: te weinig data na sanity-filter.")
            return

        # Operationele grenzen leren
        pel_min = float(d[p_el_col].quantile(0.05))
        pel_max = float(d[p_el_col].quantile(0.95))

        if mode_val == HvacMode.HEATING.value:
            self._pel_min_ufh = float(np.clip(pel_min, 0.1, 1.0))
            self._pel_max_ufh = float(np.clip(pel_max, 1.0, 4.0))
        else:
            self._pel_min_dhw = float(np.clip(pel_min, 0.1, 1.5))
            self._pel_max_dhw = float(np.clip(pel_max, 1.5, 5.0))

        t_out = d["t_out"].values.astype(float)
        t_sink = d["t_sink"].values.astype(float)
        mask = d["cop"].notna()
        t_out_c = t_out[mask]
        t_sink_c = t_sink[mask]
        cop_c = d["cop"].values[mask]

        # ── 1. Carnot-fit COP ─────────────────────────────────────────────
        # COP = eta * (t_sink + 273) / (t_sink - t_out + dT_lift)
        # Geleerde parameters: eta (efficiëntie 0-1), dT_lift (hydraulische lift K)
        # Fysische garanties:
        #   - COP daalt altijd bij hogere t_sink (want noemer groeit)
        #   - COP stijgt altijd bij hogere t_out  (want noemer krimpt)
        #   - COP > 0 altijd (bounds op dT_lift voorkomen deling door nul)

        def carnot_cop(X, eta, dT_lift):
            t_o, t_s = X
            denom = np.maximum(t_s - t_o + dT_lift, 1.0)  # nooit delen door 0
            return eta * (t_s + 273.15) / denom

        try:
            if mode_val == HvacMode.HEATING.value:
                start_values = [
                    0.45,
                    15.0,
                ]  # Lift is vaak hoger bij UFH door interne wisselaar
            else:
                start_values = [0.40, 18.0]

            popt, pcov = curve_fit(
                carnot_cop,
                (t_out_c, t_sink_c),
                cop_c,
                p0=start_values,
                bounds=([0.2, 2.0], [0.8, 30.0]),
                maxfev=10000,
            )

            eta, dT_lift = popt
            perr = np.sqrt(np.diag(pcov))

            # Kwaliteitscheck: als de fit slecht is, val terug op defaults
            cop_pred = carnot_cop((t_out_c, t_sink_c), eta, dT_lift)
            r2 = 1 - np.sum((cop_c - cop_pred) ** 2) / np.sum(
                (cop_c - cop_c.mean()) ** 2
            )

            logger.info(
                f"[PerfMap] {label} Carnot: η={eta:.3f}±{perr[0]:.3f}  "
                f"dT_lift={dT_lift:.1f}±{perr[1]:.1f}K  R²={r2:.3f}"
            )

            # We accepteren de fit als R2 > 0.35 (voor thermische data acceptabel)
            # EN als de waarden binnen fysische grenzen liggen.
            is_physically_sane = (0.25 < eta < 0.60) and (2.0 < dT_lift < 25.0)

            if r2 > 0.35 and is_physically_sane:
                logger.info(f"[PerfMap] {label} Carnot fit geaccepteerd")
            else:
                logger.warning(f"[PerfMap] {label} Carnot fit afgekeurd")
                popt = None

        except Exception as e:
            logger.warning(f"[PerfMap] {label} Carnot fit mislukt: {e}")
            popt = None

        # Fallback: typische waarden voor lucht-water WP
        if popt is None:
            if mode_val == HvacMode.HEATING.value:
                popt = (0.45, 8.0)
            else:
                popt = (0.40, 12.0)
            logger.info(
                f"[PerfMap] {label} Carnot fallback: η={popt[0]} dT_lift={popt[1]}"
            )

        if mode_val == HvacMode.HEATING.value:
            self._carnot_ufh = tuple(popt)
        else:
            self._carnot_dhw = tuple(popt)

        # ── 2. Lineaire P_el fit ──────────────────────────────────────────
        # p_el = a + b * t_sink + c * t_out
        # b >= 0: meer stroom bij hogere sinktemp (compressor harder)
        # c <= 0: minder stroom bij hogere buitentemp (vrijwel geen lift nodig)
        # Data-gedreven maar fysisch begrensd via bounds

        y_pel = d[p_el_col].values.astype(float)

        def pel_fn(X, a, b, c):
            t_o, t_s = X
            return a + b * t_s + c * t_o

        try:
            popt_pel, _ = curve_fit(
                pel_fn,
                (t_out, t_sink),
                y_pel,
                p0=[1.0, 0.01, -0.01],
                bounds=([0.2, 0.0, -0.1], [4.0, 0.05, 0.0]),
                maxfev=10000,
            )
            pel_pred = pel_fn((t_out, t_sink), *popt_pel)
            r2_pel = 1 - np.sum((y_pel - pel_pred) ** 2) / np.sum(
                (y_pel - y_pel.mean()) ** 2
            )
            logger.info(
                f"[PerfMap] {label} P_el: a={popt_pel[0]:.3f} "
                f"b={popt_pel[1]:.4f} c={popt_pel[2]:.4f}  R²={r2_pel:.3f}"
            )
        except Exception as e:
            logger.warning(f"[PerfMap] {label} P_el fit mislukt: {e}, fallback")
            if mode_val == HvacMode.HEATING.value:
                popt_pel = (0.8, 0.005, -0.01)
            else:
                popt_pel = (1.2, 0.008, -0.01)

        if mode_val == HvacMode.HEATING.value:
            self._pel_ufh = tuple(popt_pel)
        else:
            self._pel_dhw = tuple(popt_pel)

    def predict_cop(self, mode, t_out, t_sink) -> float:
        if mode == HvacMode.HEATING.value:
            params, default = self._carnot_ufh, 3.5
        else:
            params, default = self._carnot_dhw, 2.5

        if params is None:
            return default

        eta, dT_lift = params
        denom = max(t_sink - t_out + dT_lift, 1.0)
        return float(np.clip(eta * (t_sink + 273.15) / denom, 1.0, 8.0))

    def predict_pel(self, mode, t_out, t_sink) -> float:
        if mode == HvacMode.HEATING.value:
            params = self._pel_ufh
            lo, hi = 0.1, 5.0
            default = float(np.clip(1.2 - 0.02 * t_out, lo, hi))
        else:
            params = self._pel_dhw
            lo, hi = 0.1, 5.0
            default = float(np.clip(1.8 - 0.02 * t_out, lo, hi))

        if params is None:
            return default

        a, b, c = params
        # Clip ruim, zodat de helling (b) altijd zichtbaar blijft voor de PWA-tabel
        return float(np.clip(a + b * t_sink + c * t_out, lo, hi))

    def predict_p_th(self, mode: int, t_out: float, t_sink: float) -> float:
        return self.predict_pel(mode, t_out, t_sink) * self.predict_cop(
            mode, t_out, t_sink
        )


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
        self.tau_internal_gain = 1.0  # uren
        self.alpha_conv_internal = 0.5

        # ── 3-state parameters (Air – Surface – Core) ──────────────────────
        # Staat 1 T_air  (C_air):  Kamerlucht + lichte meubels + drywall [snel]
        # Staat 2 T_surf (C_surf): Vloeroppervlak ~2 cm                  [middel] ← ZON
        # Staat 3 T_core (C_core): Betonkern + UFH-buizen                 [traag]  ← WP
        self.C_air = (
            3.0  # kWh/K  — lucht + meubels + lichte wand/plafond massa (~20% van C)
        )
        self.C_surf = 3.5  # kWh/K  — vloer/wand oppervlaktelaag (~12% van C)
        self.C_core = 24.0  # kWh/K  — beton kern (rest)

        self.R_surf_air = 1.5  # K/kW  — oppervlak ↔ lucht convectie
        self.R_core_surf = 0.35  # K/kW  — kern ↔ oppervlak geleiding
        self.R_cond = 23.0  # K/kW  — wandgeleiding (constant, isolatie)
        self.R_vent = 43.0  # K/kW  — ventilatie (variabel, wind-afhankelijk)
        self.A_sol = 5.0  # m²    — effectieve zonistralings-aperture (g × A_raam)

        # ── Backward-compat afgeleide attribs (geen directe parameters meer) ─
        self.R_oa = self.R  # K/kW — totale verliesweerstand (= R_cond||R_vent)
        self.R_im = self.R_surf_air  # K/kW — proxy: massa↔lucht
        self.C_mass = self.C_surf + self.C_core  # kWh/K — totale trage massa

        self._last_internal_gain_corr = 0.0
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
            self.tau_internal_gain = d.get("tau_internal_gain", 1.0)
            self.alpha_conv_internal = d.get("alpha_conv_internal", 0.5)

            # ── 3-state parameters ────────────────────────────────────────────
            # Backward-compat: als oud model geen 3-state heeft, leid ze af
            _C_air_legacy = d.get("C_air", 3.0)
            _C_mass_legacy = d.get("C_mass", 27.0)
            _R_im_legacy = d.get("R_im", 1.2)
            _R_oa_legacy = d.get("R_oa", self.R)

            self.C_air = d.get("C_air", _C_air_legacy)
            self.C_surf = d.get("C_surf", round(0.12 * self.C, 2))
            self.C_core = d.get("C_core", max(_C_mass_legacy - self.C_surf, 5.0))
            self.R_surf_air = d.get("R_surf_air", _R_im_legacy)
            self.R_core_surf = d.get("R_core_surf", 0.35)
            self.R_cond = d.get("R_cond", _R_oa_legacy / 0.65)
            self.R_vent = d.get("R_vent", _R_oa_legacy / 0.35)
            self.A_sol = d.get("A_sol", 5.0)

            # Backward-compat afgeleide attribs
            self.R_oa = _R_oa_legacy
            self.R_im = self.R_surf_air
            self.C_mass = self.C_surf + self.C_core

            logger.info(
                f"[SysID] Geladen: R={self.R:.1f} C={self.C:.1f} "
                f"K_emit={self.K_emit:.3f} Lag={self.ufh_lag_steps * 15}m | "
                f"3-state: C_air={self.C_air:.2f} C_surf={self.C_surf:.2f} "
                f"C_core={self.C_core:.2f} kWh/K | "
                f"R_surf_air={self.R_surf_air:.3f} R_core_surf={self.R_core_surf:.3f} "
                f"R_cond={self.R_cond:.1f} R_vent={self.R_vent:.1f} K/kW | "
                f"A_sol={self.A_sol:.2f} m²"
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
        self._fit_internal_gain_tau(df_proc)
        self._fit_internal_gain_split(df_proc)
        self._fit_solar_aperture(df_proc)  # NEW: leert A_sol uit zonne-events
        self._fit_three_state()  # UPGRADE: 2→3 state

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
                "tau_internal_gain": self.tau_internal_gain,
                "alpha_conv_internal": self.alpha_conv_internal,
                # ── 3-state params ──────────────────────────────────────────
                "C_air": self.C_air,
                "C_surf": self.C_surf,
                "C_core": self.C_core,
                "R_surf_air": self.R_surf_air,
                "R_core_surf": self.R_core_surf,
                "R_cond": self.R_cond,
                "R_vent": self.R_vent,
                "A_sol": self.A_sol,
                # Backward-compat aliases
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
                f"[SysID] Vloerlag={self.ufh_lag_steps * 15} min (corr={best_corr:.2f}) "
                f"— wordt overschreven door _fit_three_state (L=0, RC-dynamica doet het werk)"
            )

    def _fit_k_emit(self, df_proc: pd.DataFrame):
        mask = (
            (df_proc["wp_ufh"] > 0.15)
            & df_proc["pure_ufh"]
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
            (df_proc["wp_dhw"] > 0.15)
            & df_proc["pure_dhw"]
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
            (df_loss["wp_dhw"] == 0)
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

    def _fit_internal_gain_tau(self, df_proc: pd.DataFrame):
        """
        Leert de thermische koppelingstijdconstante voor interne warmtebronnen.

        Fysica: Q_int(t) bereikt de kamerlucht via convectie (snel) én
        via stralingswisseling met oppervlakken (langzaam). De effectieve
        tijdconstante hangt af van de zonekoppeling, niet van het apparaat.

        Methode: cross-correlatie tussen EWM-gefilterde base_load en de
        temperatuurresidu — identiek aan _fit_floor_lag voor de vloer.
        """
        df_15 = (
            df_proc[["timestamp", "room_temp", "base_load", "wp_output", "temp"]]
            .set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna(subset=["room_temp", "base_load"])
            .reset_index()
        )

        # Verwijder periodes met actieve verwarming (domineert het signaal)
        df_15 = df_15[df_15["wp_output"] < 0.1].copy()

        # RC-model residu: wat is NIET verklaard door buiten↔binnen?
        dt = 0.25  # uur
        df_15["dT_rc"] = (
            (df_15["room_temp"] - df_15["temp"]) / self.R_oa * (dt / self.C)
        )
        df_15["dT_actual"] = df_15["room_temp"].diff()
        df_15["residual"] = df_15["dT_actual"] + df_15["dT_rc"]  # wat overblijft

        # Zoek τ die de correlatie met EWM(base_load, τ) maximaliseert
        best_tau, best_corr = 1.0, -1.0
        tau_candidates = np.arange(0.25, 8.0, 0.25)  # 15 min tot 8 uur

        for tau_h in tau_candidates:
            span = max(1, int(tau_h / 0.25))
            smoothed = df_15["base_load"].ewm(span=span).mean()
            corr = df_15["residual"].corr(smoothed)
            if pd.notna(corr) and corr > best_corr:
                best_corr, best_tau = corr, tau_h

        if best_corr < 0.05:
            logger.warning(
                f"[SysID] Geen duidelijke interne gains tijdconstante "
                f"(corr={best_corr:.3f}). Fallback τ=1.0h"
            )
            self.tau_internal_gain = 1.0
        else:
            self.tau_internal_gain = float(np.clip(best_tau, 0.25, 6.0))
            logger.info(
                f"[SysID] τ_internal_gain={self.tau_internal_gain:.2f}h "
                f"(corr={best_corr:.3f})"
            )

        self._last_internal_gain_corr = best_corr

    def _fit_internal_gain_split(self, df_proc: pd.DataFrame):
        """
        Leert welk deel van de interne gains direct naar C_air gaat (α_conv)
        versus naar C_mass via stralingsuitwisseling (1 - α_conv).

        Methode: vergelijk de impuls-response van t_room op korte pieken
        (convectief dominant, α_conv hoog) versus lange pieken (steady-state,
        split zichtbaar). Dit is de standaard-aanpak in gebouwidentificatie
        (zie: Bacher & Madsen 2011 RC-identificatie).
        """
        df = df_proc.copy()
        df = df[df["wp_output"] < 0.1].copy()  # geen verwarming actief

        df_15 = (
            df[["timestamp", "room_temp", "base_load", "temp"]]
            .set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna()
            .reset_index()
        )

        dt = 0.25
        # RC-residu: werkelijke dT minus wat het RC-model verwacht
        df_15["dT_actual"] = df_15["room_temp"].diff()
        df_15["dT_loss"] = (
            -(df_15["room_temp"] - df_15["temp"]) / self.R_oa * (dt / self.C)
        )
        df_15["residual"] = df_15["dT_actual"] - df_15["dT_loss"]

        # Piek-detector: korte pieken vs. aanhoudende load
        load = df_15["base_load"]
        df_15["load_instant"] = load  # directe convectie → C_air
        df_15["load_slow"] = load.ewm(span=8).mean()  # geïntegreerde load → C_mass

        df_fit = df_15.dropna(subset=["residual", "load_instant", "load_slow"])

        if len(df_fit) < 50:
            logger.warning(
                "[SysID] Te weinig data voor gain-split, fallback α_conv=0.5"
            )
            self.alpha_conv_internal = 0.5
            return

        # Regressie: residu = α_conv * instant/C_air + (1-α_conv) * slow/C_mass
        # Herschreven als lineair probleem:
        X = (
            np.column_stack(
                [
                    df_fit["load_instant"].values / self.C_air,
                    df_fit["load_slow"].values / self.C_mass,
                ]
            )
            * dt
        )
        y = df_fit["residual"].values

        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=0.1, fit_intercept=False, positive=True).fit(X, y)
        a, b = reg.coef_

        # Normaliseer: α_conv = aandeel naar C_air
        total = a + b
        CORR_THRESHOLD_RELIABLE = 0.25
        best_corr = self._last_internal_gain_corr

        if total > 0.01 and best_corr > CORR_THRESHOLD_RELIABLE:
            alpha = float(np.clip(a / total, 0.2, 0.8))
            logger.info(
                f"[SysID] α_conv_internal={alpha:.2f} geleerd uit data "
                f"(convectief={alpha * 100:.0f}% → C_air, "
                f"radiatief={(1 - alpha) * 100:.0f}% → C_mass, corr={best_corr:.3f})"
            )
        else:
            # Fallback: ISO 13790 prior voor zwaar gebouw met UFH
            # base_load is een slechte proxy voor ruimtelijke gains (wasmachine boven etc.)
            # Prior van 0.40 is eerlijker dan een slecht gefitte waarde
            alpha = 0.40
            logger.warning(
                f"[SysID] α_conv_internal fallback=0.40 "
                f"(corr={best_corr:.3f} < drempel {CORR_THRESHOLD_RELIABLE}, "
                f"proxy te onzuiver voor betrouwbare fit)"
            )

        self.alpha_conv_internal = alpha

    def _fit_solar_aperture(self, df_proc: pd.DataFrame):
        """
        Fit A_sol (effectieve zonistralings-aperture) uit gemeten data.

        Methode: tijdens HP-uit periodes met zonneschijn is de kamer-
        temperatuurstijging nagenoeg volledig toe te schrijven aan solar gain.
        We inverteer het RC-model en passen A_sol aan zodat:

            Q_sol = A_sol × pv_proxy  →  A_sol = C × residu / pv_proxy

        A_sol heeft hier eenheden [kW / kW_pv]: het converteert PV-stroom
        (proxy voor globale instraling) naar equivalente solar heat gain.
        Fysisch: A_sol = (A_ramen × g-waarde × oriëntatie) / (A_pv × η_pv)
        Typische waarden: 3 – 8 voor een Nederlands woonhuis.
        """
        # Selecteer beschikbare kolommen (shutter_room is optioneel)
        base_cols = ["timestamp", "room_temp", "temp", "pv_actual", "wp_output"]
        if "shutter_room" in df_proc.columns:
            base_cols.append("shutter_room")

        df_15 = (
            df_proc[base_cols]
            .set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna(subset=["room_temp", "temp", "pv_actual"])
            .reset_index()
        )

        mask = (
            (df_15["wp_output"] < 0.1)  # geen verwarming actief
            & (df_15["pv_actual"] > 0.30)  # duidelijke zonschijn (>300W)
            & (df_15["room_temp"] > 10.0)  # niet leeg/onbewoond
        )

        # Filter op open rolluiken (100 = open, 0 = gesloten)
        # Als rolluik gesloten is tijdens zon, wordt A_sol structureel onderschat.
        # We gebruiken alleen metingen waarbij het rolluik minimaal 80% open was.
        if "shutter_room" in df_15.columns:
            shutter_open = df_15["shutter_room"].fillna(100.0)
            mask = mask & (shutter_open >= 80.0)
            logger.info(
                f"[SysID] A_sol: {mask.sum()} zonne-events met open rolluiken (>=80%)"
            )
        else:
            logger.info(
                "[SysID] A_sol: geen shutter_room data — alle zonne-events gebruikt "
                "(A_sol kan onderschat zijn als rolluiken soms gesloten waren)"
            )

        d = df_15[mask].copy()

        if len(d) < 30:
            logger.warning(
                f"[SysID] A_sol: te weinig zonne-events ({len(d)}), fallback 5.0"
            )
            self.A_sol = 5.0
            return

        dt = 0.25  # uur

        # RC-gecorrigeerd residu: werkelijke dT minus verwacht RC-verlies
        d["dT_actual"] = d["room_temp"].diff()
        d["dT_loss"] = -(d["room_temp"] - d["temp"]) / self.R_oa * (dt / self.C)
        d["residual"] = d["dT_actual"] - d["dT_loss"]  # dT veroorzaakt door zon

        d = d.dropna(subset=["residual"])
        d = d[d["pv_actual"] > 0.1]  # veiligheidsdrempel (voorkom ÷0)

        # A_sol kalibratie — fysisch correct [kW/kW]
        #
        # Enkelvoudig RC-model: C × dT_air = [A_sol × pv - (T_air-T_out)/R_oa] × dt
        # → residual = A_sol × pv × dt / C  [K per stap]
        # → A_sol [kW/kW] = C × residual / (pv × dt)
        #
        # A_sol is een fysische eigenschap van het gebouw (g-waarde × raamoppervlak
        # gedeeld door PV-oppervlak × η_pv). Het is ONAFHANKELIJK van welke
        # thermische node de warmte opneemt.
        #
        # In de MPC wordt A_sol gebruikt als:
        #   gain_solar [kW] = A_sol [kW/kW] × pv [kW] × shutter_frac [-]
        # Dit is vermogen [kW]. In T_surf-dynamica: dT_surf = gain_solar × dt / C_surf
        # wat dimensioneel klopt: kW × h / (kWh/K) = K ✓
        #
        # Typische waarden voor een Nederlands woonhuis: 1–10 kW/kW
        ratio = self.C * d["residual"] / (d["pv_actual"] * dt)

        # Gebruik 65e percentiel: robuuster dan gemiddelde, negeert bewolkte outliers
        a_sol_est = float(ratio.quantile(0.65))

        self.A_sol = float(np.clip(a_sol_est, 1.0, 15.0))
        logger.info(
            f"[SysID] A_sol={self.A_sol:.2f} geleerd uit "
            f"{len(d)} zonne-events (65e percentiel)"
        )

    def _fit_three_state(self):
        """
        3-state parameterschatting (Air – Surface – Core) — Gold Standard.

        Staat 1  T_air  (C_air):  Kamerlucht + lichte meubels         [snel]
        Staat 2  T_surf (C_surf): Vloer/wand oppervlaktelaag ~2 cm    [middel] ← ZON
        Staat 3  T_core (C_core): Betonkern met UFH-buizen             [traag]  ← WP

        Thermische koppelingen:
          R_surf_air  — convectie oppervlak → lucht   (snel)
          R_core_surf — geleiding kern → oppervlak    (traag)
          R_cond      — wandgeleiding lucht → buiten  (constant)
          R_vent      — ventilatie lucht → buiten     (variabel met wind)

        Fysica:
          dT_air /dt = [(T_surf - T_air)/R_sa - (T_air - T_out)/R_oa] / C_air
          dT_surf/dt = [Q_sol + (T_core - T_surf)/R_cs - (T_surf - T_air)/R_sa] / C_surf
          dT_core/dt = [P_th  - (T_core - T_surf)/R_cs]                          / C_core

        waarbij R_oa = R_cond || R_vent (parallel pad).
        """

        # ── 1. Totale verliesweerstand (geleerd via _fit_rc) ─────────────────
        self.R_oa = self.R  # backward-compat

        # Splits R_oa in parallel paden R_cond en R_vent
        # 1/R_oa = 1/R_cond + 1/R_vent
        # Aandeel ventilatie voor goed geïsoleerde woning: ~35 %
        F_VENT = 0.35
        self.R_cond = float(np.clip(self.R / (1.0 - F_VENT), 5.0, 150.0))
        self.R_vent = float(np.clip(self.R / F_VENT, 5.0, 200.0))

        # ── 2. Thermische massa opsplitsen ────────────────────────────────────
        # Zware bouw (beton/UFH):
        #   C_air  ≈ 20% van C — lucht + meubels + lichte drywall-laag die direct
        #                         met de lucht uitwisselt. NIET alleen lucht!
        #                         ISO 13790 voor zwaar gebouw: ~18-25%.
        #                         Fysisch: met C_air te klein wordt T_air VEEL te
        #                         gevoelig voor directe gains (interne lasten → T_air).
        #   C_surf ≈ 12% van C — vloer/wand oppervlaktelaag ~2 cm beton
        #   C_core ≈ 68% van C — diepe betonkern met UFH-buizen
        C_AIR_FRAC = 0.20  # 20% → lucht + alle "lichte" massa (meubels, drywall)
        C_SURF_FRAC = 0.12  # 12% → vloer/wand oppervlaktelaag

        c_air_lower = max(1.0, 0.10 * self.C)
        c_air_upper = 0.28 * self.C
        self.C_air = float(np.clip(C_AIR_FRAC * self.C, c_air_lower, c_air_upper))

        self.C_surf = float(np.clip(C_SURF_FRAC * self.C, 0.5, 0.18 * self.C))
        self.C_core = float(max(self.C - self.C_air - self.C_surf, 5.0))

        # ── 3. R_surf_air: oppervlak → lucht convectie ───────────────────────
        # Fysisch: R = 1 / (h_eff × A_floor)
        # h_eff ≈ 8 W/m²K (convectie + straling vloer), A_floor ≈ 80–100 m²
        # → R ≈ 1000/(8×80) ≈ 1.56 K/kW
        # Gerelateerd aan K_emit: R_emit = 1/K_emit is het totale UFH-pad.
        # Oppervlak→lucht is ~60 % van dat totale weerstandsdeel.
        R_emit = 1.0 / max(self.K_emit, 0.05)
        self.R_surf_air = float(np.clip(0.6 * R_emit, 0.3, 6.0))

        # ── 4. R_core_surf: kern → oppervlak geleiding ───────────────────────
        # Fysisch: R = d / (λ × A_floor)
        # d ≈ 50 mm, λ(beton) ≈ 1.7 W/m·K, A ≈ 80 m²
        # → R_physical ≈ 0.05/(1.7×80) × 1000 = 0.368 K/kW
        R_physical = 0.37

        # Data-gestuurd: τ_lag ≈ R_core_surf × C_core
        tau_lag_h = (self.ufh_lag_steps * 15) / 60.0  # uren
        R_from_lag = tau_lag_h / max(self.C_core, 5.0)

        # Blend: evengewicht tussen fysische prior en data-gestuurde schatting
        self.R_core_surf = float(np.clip(0.5 * (R_from_lag + R_physical), 0.05, 2.0))

        # ── 5. Backward-compat afgeleide attribs ─────────────────────────────
        self.R_im = self.R_surf_air  # proxy: oppervlak↔lucht was massa↔lucht
        self.C_mass = self.C_surf + self.C_core  # totale trage massa

        # ── 6. UFH-lag override voor 3-state model ────────────────────────────
        # In het 2-state model was ufh_lag_steps nodig om de vloer-traagheid
        # te compenseren die het model niet kende.
        # In het 3-state model wordt die vertraging al VOLLEDIG gevangen door
        # de RC-tijdconstanten:
        #   τ_kern→opp  = R_core_surf × C_core ≈ 128 min
        #   τ_lucht_volgt_opp = R_surf_air × C_air ≈ 127 min
        # Een extra kunstmatige P_th-vertraging (L = ufh_lag_steps) telt de
        # vertraging dubbel: de MPC denkt dat UFH nog later werkt dan de
        # RC-dynamica al impliceert → optimizer start te vroeg/verkeerd.
        # Oplossing: L = 0; de RC-structuur doet het werk.
        self.ufh_lag_steps = 0

        # ── 7. Consistentiecheck & logging ───────────────────────────────────
        tau_surf_air_min = self.R_surf_air * self.C_surf * 60  # minuten (C_surf-node)
        tau_air_track_min = (
            self.R_surf_air * self.C_air * 60
        )  # minuten (T_air volgt T_surf)
        tau_core_surf_min = self.R_core_surf * self.C_core * 60  # minuten

        logger.info(
            f"[SysID] 3-state: "
            f"C_air={self.C_air:.2f}  C_surf={self.C_surf:.2f}  "
            f"C_core={self.C_core:.2f} kWh/K | "
            f"R_surf_air={self.R_surf_air:.3f}  R_core_surf={self.R_core_surf:.3f}  "
            f"R_cond={self.R_cond:.1f}  R_vent={self.R_vent:.1f}  "
            f"R_oa={self.R_oa:.2f} K/kW | "
            f"A_sol={self.A_sol:.2f} kW/kW | "
            f"τ_lucht_volgt_opp={tau_air_track_min:.0f} min  "
            f"τ_opp_node={tau_surf_air_min:.0f} min  "
            f"τ_kern→opp={tau_core_surf_min:.0f} min  "
            f"(vloerlag=0 in 3-state MPC — RC-dynamica vangt dit al)"
        )

        if R_from_lag > 2.0:
            logger.warning(
                f"[SysID] R_core_surf uit lag ({R_from_lag:.3f} K/kW) > 2.0. "
                f"Vloer-lag en C_core zijn inconsistent → blend met fysische prior gebruikt."
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
            (df["wp_ufh"] > 0.15)
            & df["steady_state"]
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
            (df["wp_dhw"] > 0.15)
            & df["steady_state"]
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
    def __init__(self, path, R, C, tau_internal: float = 1.0, alpha_conv: float = 0.5):
        self.path = Path(path)
        self.R = R
        self.C = C
        self.tau_internal = tau_internal
        self.alpha_conv = alpha_conv
        self.a_sol = (
            5.0  # effectieve zonistralings-aperture [kW/kW_pv] — gesynchroniseerd
        )
        # vanuit SystemIdentificator.A_sol na training
        self.model = None
        self.is_fitted = False
        self.features = [
            "temp",
            "solar",
            "effective_solar",
            "shutter_room",
            "internal_load",
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

    def _get_smoothed_load(self, base_load_series: pd.Series) -> pd.Series:
        # Gebruik de geleerde tau voor het laagdoorlaatfilter
        span = max(1, int(self.tau_internal / 0.125))
        return base_load_series.ewm(span=span).mean()

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

        is_heating = df_proc["pure_ufh"] & (df_proc["wp_ufh"] > 0.15)
        just_stopped = is_heating.shift(1, fill_value=False) & ~is_heating
        just_started = ~is_heating.shift(1, fill_value=True) & is_heating

        df_proc["post_heat_cooldown"] = (
            just_stopped.rolling(window=4, min_periods=1).max().fillna(0).astype(bool)
        )
        df_proc["warmup_transient"] = (
            just_started.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)
        )

        dt_hours = df_proc["timestamp"].diff().dt.total_seconds().shift(-1) / 3600
        mask = (
            (is_heating | (df_proc["wp_output"] < 0.1))
            & ~df_proc["post_heat_cooldown"]
            & ~df_proc["warmup_transient"]
        )
        df_feat = df_proc[mask].copy()
        dt = dt_hours[mask]

        t_curr = df_feat["room_temp"]
        t_next = df_feat["room_temp"].shift(-1)
        p_heat = df_feat["wp_output"]
        t_out = df_feat["temp"]
        t_model_next = t_curr + ((p_heat - (t_curr - t_out) / self.R) * dt / self.C)
        target = (t_next - t_model_next) * self.C / dt

        df_feat = add_cyclic_time_features(df_feat, "timestamp")
        df_feat["solar"] = df_feat["pv_actual"]
        df_feat["target"] = target
        df_feat["effective_solar"] = df_feat["solar"] * (
            df_feat.get("shutter_room", 100) / 100.0
        )
        # internal_load: basisverbruik als proxy voor bezetting/interne warmte
        df_feat["internal_load"] = self._get_smoothed_load(df_feat["base_load"])

        train_set = df_feat[self.features + ["target"]].dropna()

        q_low = train_set["target"].quantile(0.01)
        q_high = train_set["target"].quantile(0.99)
        train_set = train_set[train_set["target"].between(q_low, q_high)]

        if len(train_set) > 10:
            self.model = Ridge(alpha=0.1).fit(
                train_set[self.features], train_set["target"]
            )
            self.is_fitted = True
            joblib.dump(self.model, self.path)
            logger.info("[UFH Residual] Getraind.")

    def predict(self, forecast_df, shutters):
        if forecast_df is None:
            return np.array([]), np.array([]), np.array([])

        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        solar_raw = np.asarray(
            df.get("power_corrected", df.get("pv_estimate", 0.0))
        ).astype(float)
        shutter_frac = np.clip(np.asarray(shutters, dtype=float) / 100.0, 0.0, 1.0)

        # ── Zonopwarming: ALTIJD via A_sol × G_rad_proxy × shutter ──────────
        #
        # Fysische formule: Q_sol = A_sol [kW/kW_pv] × pv_proxy [kW] × f_shutter [-]
        #
        # A_sol is geleerd door SystemIdentificator._fit_solar_aperture():
        #   het converteert PV-stroom naar equivalente solar heat gain.
        #
        # shutter_frac:
        #   0.0 = volledig gesloten  → Q_sol = 0   (geen instraling)
        #   0.5 = half gesloten      → Q_sol halveert
        #   1.0 = volledig open      → volledige instraling
        #
        # Dit is de ENIGE plek waar shutters de zongain beïnvloeden.
        # Ridge-model wordt NIET meer gebruikt voor zon (was een proxy, nu fysica).
        # ─────────────────────────────────────────────────────────────────────
        gain_solar = np.maximum(self.a_sol * solar_raw * shutter_frac, 0.0)

        if self.model is None or not self.is_fitted:
            # Zonder Ridge-model: alleen solar-gains beschikbaar
            n = len(df)
            gain_air = np.zeros(n)
            gain_surf = gain_solar
            gain_core = np.zeros(n)
            return gain_air, gain_surf, gain_core

        df["solar"] = solar_raw
        df["shutter_room"] = shutters
        df["effective_solar"] = solar_raw * shutter_frac  # bewaard voor Ridge (intern)
        df["internal_load"] = self._get_smoothed_load(df["load_corrected"])

        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.features].values
        coef = self.model.coef_  # Ridge coëfficiënten, shape (n_features,)

        # Interne gains (apparaten, mensen) — uit Ridge model
        idx_internal = self.features.index("internal_load")
        gain_internal = np.maximum(X[:, idx_internal] * coef[idx_internal], 0.0)

        # ── 3-state gain split (fysisch correct, 3C2R model) ────────────────
        #
        # ZON:       → volledig naar T_surf (ramen → vloeroppervlak absorptie)
        #              T_surf → T_air via convectie (R_surf_air)
        #              Effect van SHUTTER: expliciet via shutter_frac hierboven
        #
        # INTERN:    → alpha_conv-deel direct naar T_air (convectief: lampen, computers)
        #              (1-alpha_conv)-deel naar T_surf (radiatief: TV, vloer-warmte)
        #
        # KERN:      → geen directe externe bron, alleen WP via UFH-buizen
        # ─────────────────────────────────────────────────────────────────────
        gain_air = gain_internal * self.alpha_conv
        gain_surf = gain_solar + gain_internal * (1.0 - self.alpha_conv)
        gain_core = np.zeros_like(gain_air)

        return gain_air, gain_surf, gain_core


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
        df = clean_thermal_data(df).copy().sort_values("timestamp")

        # Detectie: gebruik dhw_top (gevoelig voor onttrekking)
        df["dhw_diff_top"] = df["dhw_top"].diff()

        # Kwantificering: gebruik gemiddelde (consistent met MPC state-variabele)
        df["dhw_avg"] = (df["dhw_top"] + df["dhw_bottom"]) / 2.0
        df["dhw_diff_avg"] = df["dhw_avg"].diff()

        # Markeer post-DHW stratificatieperiode
        is_dhw = df["pure_dhw"] & (df["wp_dhw"] > 0.15)
        just_stopped_dhw = is_dhw.shift(1, fill_value=False) & ~is_dhw
        df["post_dhw_stratification"] = (
            just_stopped_dhw.rolling(window=4, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )

        mask_shower = (
            (df["wp_dhw"] == 0)
            & (df["dhw_diff_top"] < -0.5)
            & (df["dhw_diff_avg"] < 0.0)
            & ~df["post_dhw_stratification"]
        )
        df["demand"] = 0.0
        df.loc[mask_shower, "demand"] = df["dhw_diff_avg"].abs()
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
        if forecast_df is None:
            return np.array([])

        if self.model is None or not self.is_fitted:
            return np.zeros(len(forecast_df))

        df_feat = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        predictions = self.model.predict(df_feat[self.features])
        return np.where(predictions < 0.8, 0.0, predictions)


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
        export_price: float,
    ):
        self.C_room = C_room
        self.C_tank = C_tank
        self.avg_cop_ufh = avg_cop_ufh
        self.avg_cop_dhw = avg_cop_dhw
        self.export_price = export_price

    def compute(self, max_price: float) -> dict:
        # Kosten om 1K te herstellen op het duurste moment
        # Kosten om 1K te herstellen (voor ondergrenzen)
        cost_to_heat_1k_room = (self.C_room / self.avg_cop_ufh) * max_price
        cost_to_heat_1k_tank = (self.C_tank / self.avg_cop_dhw) * max_price

        # ONDER: Gebruik max_price (comfort is prioriteit)
        room_under = cost_to_heat_1k_room * 1.01
        tank_under = cost_to_heat_1k_tank * 1.01

        # OVER: De waarde van de energie is wat het oplevert bij export!
        # Dit is de 'opportunity cost'.
        room_over = (self.C_room / self.avg_cop_ufh) * self.export_price
        tank_over = (self.C_tank / self.avg_cop_dhw) * self.export_price

        # Terminal waarden
        terminal_room = cost_to_heat_1k_room
        terminal_tank = cost_to_heat_1k_tank

        logger.info(
            f"[ComfortCost] max_prijs={max_price:.3f} | "
            f"kamer_onder={room_under:.4f} over={room_over:.4f} | "
            f"tank_onder={tank_under:.4f} over={tank_over:.4f}"
        )

        return {
            "room_under": room_under,
            "room_over": room_over,
            "tank_under": tank_under,
            "tank_over": tank_over,
            "terminal_room": terminal_room,
            "terminal_tank": terminal_tank,
        }


class PWATable:
    def __init__(self, perf_map: HPPerformanceMap, hydraulic: HydraulicPredictor):
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.t_out_grid = np.arange(-10.0, 18.0, 2.0)
        self.t_sink_ufh = np.arange(16.0, 24.0, 1.0)
        self.t_sink_dhw = np.arange(10.0, 70.0, 5.0)
        self._build()

    def _build(self):
        T_o = self.t_out_grid
        self.G_ufh = np.zeros((len(T_o), len(self.t_sink_ufh)))
        self.C_ufh = np.zeros_like(self.G_ufh)
        self.G_dhw = np.zeros((len(T_o), len(self.t_sink_dhw)))
        self.C_dhw = np.zeros_like(self.G_dhw)
        self.S_ufh = np.zeros_like(self.G_ufh)
        self.S_dhw = np.zeros_like(self.G_dhw)

        for i, t_out in enumerate(T_o):
            for j, t_sink in enumerate(self.t_sink_ufh):
                p_th = self.perf_map.predict_p_th(HvacMode.HEATING.value, t_out, t_sink)
                sup = self.hydraulic.predict_supply("UFH", p_th, t_out, t_sink)
                self.G_ufh[i, j] = self.perf_map.predict_pel(
                    HvacMode.HEATING.value, t_out, t_sink
                )
                self.C_ufh[i, j] = self.perf_map.predict_cop(
                    HvacMode.HEATING.value, t_out, t_sink
                )
                self.S_ufh[i, j] = sup

            for j, t_sink in enumerate(self.t_sink_dhw):
                p_th = self.perf_map.predict_p_th(HvacMode.DHW.value, t_out, t_sink)
                sup = self.hydraulic.predict_supply("DHW", p_th, t_out, t_sink)
                self.G_dhw[i, j] = self.perf_map.predict_pel(
                    HvacMode.DHW.value, t_out, t_sink
                )
                self.C_dhw[i, j] = self.perf_map.predict_cop(
                    HvacMode.DHW.value, t_out, t_sink
                )
                self.S_dhw[i, j] = sup

        logger.info(
            f"[PWA] Grid gebouwd: UFH {self.G_ufh.shape}  DHW {self.G_dhw.shape}"
        )

        # Zoek de index die het dichtst bij 7 graden buiten ligt
        idx_7deg = np.abs(self.t_out_grid - 7.0).argmin()

        logger.debug(f"[PWA] DHW Inspectie bij {self.t_out_grid[idx_7deg]}°C buiten")
        for j, t_s in enumerate(self.t_sink_dhw):
            pel = self.G_dhw[idx_7deg, j]
            cop = self.C_dhw[idx_7deg, j]
            logger.info(f"Tank={t_s:4.1f}°C | P_el={pel:.3f}kW | COP={cop:.2f}")

    def _interp2d(self, t_out, t_sink, t_sink_grid, G, C, S):
        t_o = np.clip(t_out, self.t_out_grid[0], self.t_out_grid[-1])
        t_s = np.clip(t_sink, t_sink_grid[0], t_sink_grid[-1])
        n = len(t_sink_grid)
        pel = np.interp(
            t_s,
            t_sink_grid,
            [float(np.interp(t_o, self.t_out_grid, G[:, j])) for j in range(n)],
        )
        cop = np.interp(
            t_s,
            t_sink_grid,
            [float(np.interp(t_o, self.t_out_grid, C[:, j])) for j in range(n)],
        )
        sup = np.interp(
            t_s,
            t_sink_grid,
            [float(np.interp(t_o, self.t_out_grid, S[:, j])) for j in range(n)],
        )
        return float(pel), float(cop), float(sup)

    def compute(
        self,
        t_out_arr: np.ndarray,
        t_room_arr: np.ndarray,
        t_dhw_arr: np.ndarray,
    ) -> dict:
        T = len(t_out_arr)

        # Arrays voor de lineaire (Taylor) coëfficiënten
        pel_const_ufh = np.zeros(T)
        pel_slope_ufh = np.zeros(T)
        pth_const_ufh = np.zeros(T)
        pth_slope_ufh = np.zeros(T)

        pel_const_dhw = np.zeros(T)
        pel_slope_dhw = np.zeros(T)
        pth_const_dhw = np.zeros(T)
        pth_slope_dhw = np.zeros(T)

        sup_ufh = np.zeros(T)
        sup_dhw = np.zeros(T)
        avg_cop_ufh = np.zeros(T)
        avg_cop_dhw = np.zeros(T)

        for t in range(T):
            # --- UFH Berekening ---
            t_out = t_out_arr[t]
            t_room = t_room_arr[t]

            # Base operating point
            pel_u, cop_u, sup_u = self._interp2d(
                t_out, t_room, self.t_sink_ufh, self.G_ufh, self.C_ufh, self.S_ufh
            )
            pth_u = pel_u * cop_u

            # Perturbatie (+1K) om numerieke afgeleide (slope) te bepalen
            pel_u_plus, cop_u_plus, _ = self._interp2d(
                t_out, t_room + 1.0, self.t_sink_ufh, self.G_ufh, self.C_ufh, self.S_ufh
            )
            pth_u_plus = pel_u_plus * cop_u_plus

            # Hellingen (d/dT)
            dpel_dt_u = pel_u_plus - pel_u
            dpth_dt_u = pth_u_plus - pth_u

            # Taylor intercept: y - m*x
            pel_const_ufh[t] = pel_u - dpel_dt_u * t_room
            pth_const_ufh[t] = pth_u - dpth_dt_u * t_room
            pel_slope_ufh[t] = dpel_dt_u
            pth_slope_ufh[t] = dpth_dt_u

            sup_ufh[t] = sup_u
            avg_cop_ufh[t] = cop_u

            # --- DHW Berekening ---
            t_dhw = t_dhw_arr[t]
            pel_d, cop_d, sup_d = self._interp2d(
                t_out, t_dhw, self.t_sink_dhw, self.G_dhw, self.C_dhw, self.S_dhw
            )
            pth_d = pel_d * cop_d

            pel_d_plus, cop_d_plus, _ = self._interp2d(
                t_out, t_dhw + 1.0, self.t_sink_dhw, self.G_dhw, self.C_dhw, self.S_dhw
            )
            pth_d_plus = pel_d_plus * cop_d_plus

            dpel_dt_d = pel_d_plus - pel_d
            dpth_dt_d = pth_d_plus - pth_d

            pel_const_dhw[t] = pel_d - dpel_dt_d * t_dhw
            pth_const_dhw[t] = pth_d - dpth_dt_d * t_dhw
            pel_slope_dhw[t] = dpel_dt_d
            pth_slope_dhw[t] = dpth_dt_d

            sup_dhw[t] = sup_d
            avg_cop_dhw[t] = cop_d

            if t == 0:
                logger.debug(
                    f"[PWA] T_dhw={t_dhw:.1f} | "
                    f"Pel_base={pel_d:.3f} | Pel_plus={pel_d_plus:.3f} | "
                    f"SLOPE={dpel_dt_d:.4f}"
                )

        return {
            "pel_const_ufh": pel_const_ufh,
            "pel_slope_ufh": pel_slope_ufh,
            "pth_const_ufh": pth_const_ufh,
            "pth_slope_ufh": pth_slope_ufh,
            "pel_const_dhw": pel_const_dhw,
            "pel_slope_dhw": pel_slope_dhw,
            "pth_const_dhw": pth_const_dhw,
            "pth_slope_dhw": pth_slope_dhw,
            "sup_ufh": sup_ufh,
            "sup_dhw": sup_dhw,
            "avg_cop_ufh": avg_cop_ufh,
            "avg_cop_dhw": avg_cop_dhw,
        }

    def rebuild(self):
        """Aanroepen na elke train()-cyclus."""
        self._build()


class ThermalEKF:
    """
    3-state Kalman Filter: [T_air, T_surf, T_core].

    Staat 1  T_air  (C_air):  Kamerlucht + lichte meubels          [snel, gemeten]
    Staat 2  T_surf (C_surf): Vloer/wand oppervlaktelaag            [middel, ZON hier]
    Staat 3  T_core (C_core): Betonkern + UFH-buizen                [traag, WP hier]

    Thermisch model (voorwaartse Euler voor predict, Kalman update op T_air):

        dT_air /dt = [(T_surf - T_air)/R_sa - (T_air - T_out)/R_oa + q_air ] / C_air
        dT_surf/dt = [Q_sol + (T_core-T_surf)/R_cs - (T_surf-T_air)/R_sa + q_surf] / C_surf
        dT_core/dt = [P_th  - (T_core - T_surf)/R_cs                + q_core] / C_core

    R_oa = R_cond || R_vent (parallel verlies lucht → buiten)
    """

    def __init__(self, ident):
        dt = 0.25  # uur (kwartier-stap)

        C_air = ident.C_air
        C_surf = ident.C_surf
        C_core = ident.C_core
        R_sa = ident.R_surf_air  # surface ↔ air
        R_cs = ident.R_core_surf  # core   ↔ surface
        R_oa = ident.R_oa  # air    → outside (= R_cond || R_vent)

        # ── State-transitiematrix A (3×3) ─────────────────────────────────
        # Rij 0: T_air  verliest naar buiten (R_oa) en uitwisselt met T_surf (R_sa)
        # Rij 1: T_surf uitwisselt met T_air (R_sa) en T_core (R_cs)
        # Rij 2: T_core verliest naar T_surf (R_cs), krijgt van WP
        a11 = 1.0 - dt / (R_sa * C_air) - dt / (R_oa * C_air)
        a12 = dt / (R_sa * C_air)
        a13 = 0.0

        a21 = dt / (R_sa * C_surf)
        a22 = 1.0 - dt / (R_cs * C_surf) - dt / (R_sa * C_surf)
        a23 = dt / (R_cs * C_surf)

        a31 = 0.0
        a32 = dt / (R_cs * C_core)
        a33 = 1.0 - dt / (R_cs * C_core)

        self.A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

        # ── Meetmatrix: alleen T_air gemeten ──────────────────────────────
        self.H = np.array([[1.0, 0.0, 0.0]])

        # ── Procesruis (Q): T_core verandert het traagst ──────────────────
        self.Q = np.diag([0.010, 0.005, 0.001])

        # ── Meetruis (R): thermostaat ±0.2 K → var = 0.04 ────────────────
        self.R_n = np.array([[0.04]])

        self.P = np.eye(3) * 2.0
        self.ident = ident
        self.x = None
        self._dt = dt

    def reset(self, t_air: float, t_surf: float, t_core: float):
        self.x = np.array([t_air, t_surf, t_core])
        self.P = np.eye(3) * 2.0

    def predict_step(
        self,
        p_heat: float,
        t_out: float,
        gain_air: float,
        gain_surf: float,
        gain_core: float,
    ):
        """
        Voorspel één stap vooruit.

        p_heat    : thermisch vermogen WP naar T_core [kW]
        t_out     : buitentemperatuur [°C]
        gain_air  : convectieve interne gains → T_air [kW]
        gain_surf : ZON + radiatieve interne gains → T_surf [kW]
        gain_core : directe gains → T_core [kW] (normaal 0)
        """
        if self.x is None:
            return

        dt = self._dt
        C_air = self.ident.C_air
        C_surf = self.ident.C_surf
        C_core = self.ident.C_core
        R_oa = self.ident.R_oa

        # Externe input-vector (niet in A):
        # T_out drijft T_air via R_oa; gains en P_th gaan naar respectieve staten
        f_ext = np.array(
            [
                t_out * dt / (R_oa * C_air) + gain_air * dt / C_air,
                gain_surf * dt / C_surf,
                p_heat * dt / C_core + gain_core * dt / C_core,
            ]
        )

        self.x = self.A @ self.x + f_ext
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, t_air_meas: float) -> tuple[float, float, float]:
        """
        Kalman-correctie op T_air meting.
        Geeft gecorrigeerde (T_air, T_surf, T_core) terug.
        """
        if self.x is None:
            return t_air_meas, t_air_meas, t_air_meas

        # Innovatie-covariantie (1×1)
        S = self.H @ self.P @ self.H.T + self.R_n

        # Kalman gain (3×1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovatie
        expected_meas = (self.H @ self.x).item()
        innov = t_air_meas - expected_meas

        # State en covariantie bijwerken
        self.x += K.flatten() * innov
        self.P = (np.eye(3) - K @ self.H) @ self.P

        logger.info(
            f"[EKF] K={K.flatten().round(3)}  innov={innov:.3f}K  "
            f"T_air={self.x[0]:.2f}  T_surf={self.x[1]:.2f}  T_core={self.x[2]:.2f}"
        )
        return float(self.x[0]), float(self.x[1]), float(self.x[2])
