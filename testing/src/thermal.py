import pandas as pd
import numpy as np
import joblib
import logging

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils import add_cyclic_time_features
from context import HvacMode

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


def clean_thermal_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verwijdert fysisch corrupte rijen uit de meetdata.
    Gooit alleen weg wat aantoonbaar fout is — geen schattingen, geen imputatie.

    Regels:
    1. Onbekende HVAC modes (niet 0/1/2) worden gefilterd.
    2. wp_actual < 0 is fysisch onmogelijk (meetruis meter).
    3. Rijen zonder return_temp worden gefilterd: delta_t is dan niet te berekenen.
    4. Negatieve delta_t (supply < return) is fysisch onmogelijk bij normale werking.
    5. DHW-opstarttransienten: supply kouder dan de tank betekent geen warmteoverdracht.
    6. UFH-transienten na DHW: supply > 45°C in UFH-mode is resterende DHW-vloeistof.
    7. dhw_top < dhw_bottom: warmwater stijgt op, inversie wijst op sensor- of logfout.
    8. wp_actual aanwezig maar < 0.15 kW terwijl HVAC actief: compressor draait niet echt.
    """
    df = df.copy()

    # Zorg dat alle thermische kolommen numeriek zijn
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

    # 1. Filter onbekende HVAC modes (bijv. mode 4 = koelen/defrost)
    df = df[
        df["hvac_mode"].isin(
            [
                HvacMode.OFF.value,
                HvacMode.HEATING.value,
                HvacMode.DHW.value,
            ]
        )
    ]

    # 2. Negatief elektrisch vermogen is fysisch onmogelijk
    df = df[df["wp_actual"].isna() | (df["wp_actual"] >= 0.0)]

    # 3. Bereken delta_t; rijen zonder return_temp vallen af
    df["delta_t"] = df["supply_temp"] - df["return_temp"]
    # Rijen zonder return_temp hebben delta_t=NaN -> laat ze staan voor standby-analyse,
    # maar markeer ze zodat klassen die delta_t nodig hebben ze kunnen droppen.
    # (Standby mode=0 heeft legitiem geen supply/return; die rijen zijn nuttig voor R/C training.)

    # 4. Negatieve delta_t bij actieve modus is fysisch onmogelijk
    active = df["hvac_mode"].isin([HvacMode.HEATING.value, HvacMode.DHW.value])
    corrupt_delta = active & df["delta_t"].notna() & (df["delta_t"] < 0.0)
    df = df[~corrupt_delta]

    # 5. DHW-opstarttransienten: supply kouder dan tank = compressor heeft nog niet geleverd
    mask_dhw = df["hvac_mode"] == HvacMode.DHW.value
    dhw_cold_start = mask_dhw & (df["supply_temp"] < df["dhw_bottom"] + 1.0)
    df = df[~dhw_cold_start]

    # 6. UFH-transienten na DHW-run: supply > 30°C in UFH-mode = resterende DHW-vloeistof in leiding
    mask_ufh = df["hvac_mode"] == HvacMode.HEATING.value
    ufh_hot_transient = mask_ufh & (df["supply_temp"] > 30.0)
    df = df[~ufh_hot_transient]

    # 7. dhw_top < dhw_bottom: fysisch onmogelijk (warmwater stijgt op)
    if "dhw_top" in df.columns and "dhw_bottom" in df.columns:
        corrupt_dhw = (
            df["dhw_top"].notna()
            & df["dhw_bottom"].notna()
            & (df["dhw_top"] < df["dhw_bottom"])
        )
        df = df[~corrupt_dhw]

    # 8. Actieve modus maar wp_actual is sluipverbruik (< 0.15 kW): compressor draait niet
    # Herbereken active op de huidig gefilterde df zodat de index overeenkomt
    active = df["hvac_mode"].isin([HvacMode.HEATING.value, HvacMode.DHW.value])
    standby_noise = active & df["wp_actual"].notna() & (df["wp_actual"] < 0.15)
    df = df[~standby_noise]

    # Filter deelkwartieren: wp_actual moet dicht bij steady-state liggen
    # Gebruik een ondergrens van 80% van de mediaan per modus
    # zodat opstarts en stops niet mee-trainen
    for mode_val in [HvacMode.HEATING.value, HvacMode.DHW.value]:
        mask = df["hvac_mode"] == mode_val
        if mask.sum() > 10:
            median_wp = df.loc[mask, "wp_actual"].median()
            # Rijen onder 70% van mediaan zijn deelkwartieren
            partial = mask & (df["wp_actual"] < 0.70 * median_wp)
            df = df[~partial]

    # 9. Markeer volledige kwartieren: de vorige én volgende rij hebben dezelfde HVAC-modus.
    # Een deelkwartier (WP start of stopt halverwege) heeft een verkeerde verhouding
    # tussen wp_actual en temperatuurverandering. Gebruik full_quarter als filter-kolom
    # in trainingsmethodes die gevoelig zijn voor deze verhouding (K_emit, K_tank, DHW stooklijn).
    df["full_quarter"] = (df["hvac_mode"] == df["hvac_mode"].shift(1)) & (
        df["hvac_mode"] == df["hvac_mode"].shift(-1)
    )

    return df


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
                logger.info("[Thermal] Performance map geladen.")
            except Exception as e:
                logger.warning(f"[Thermal] Performance map laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = clean_thermal_data(df)

        # ============================
        # 1. Fysisch thermisch vermogen berekenen
        # ============================

        # Na clean_thermal_data zijn rijen zonder return_temp al gefilterd voor actieve modes,
        # maar droppen we hier expliciet voor de COP-berekening die delta_t vereist.
        df = df.dropna(subset=["delta_t", "wp_actual", "supply_temp", "return_temp"])
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

        mask = (df["cop"].between(1.0, 8.0)) & (df["delta_t"].between(1.0, 15.0))
        df_clean = df[mask].copy()

        if len(df_clean) < 50:
            logger.warning(
                "[Thermal] Te weinig valide data voor performance map training."
            )
            return

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

        # Wat is het maximale en minimale elektrische vermogen per buitentemperatuur?
        max_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.99).reset_index()
        min_stats = df_f.groupby("t_rounded")["wp_actual"].quantile(0.05).reset_index()

        if not max_stats.empty:
            self.max_pel_model = LinearRegression().fit(
                max_stats[["t_rounded"]], max_stats["wp_actual"]
            )
            self.min_pel_model = LinearRegression().fit(
                min_stats[["t_rounded"]], min_stats["wp_actual"]
            )

        self.is_fitted = True
        joblib.dump(
            {
                "cop_model": self.cop_model,
                "max_pel_model": self.max_pel_model,
                "min_pel_model": self.min_pel_model,
            },
            self.path,
        )
        logger.info("[Thermal] Performance Map getraind.")

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


# =========================================================
# 2. SYSTEM IDENTIFICATOR (MET TRAAGHEID)
# =========================================================
class SystemIdentificator:
    def __init__(self, path, tank_liters):
        self.path = Path(path)
        # C_tank: gebruik de fysische waarde op basis van tankinhoud.
        # De integraal-methode is onbetrouwbaar omdat:
        # 1. dhw_bottom meet de koudste plek van een gelaagde tank, niet het gemiddelde
        # 2. Eerste en laatste kwartier van een run zijn deelkwartieren met verstoorde
        #    energie/temperatuur-verhouding
        # Bij een bekende tankinhoud is de fysische berekening nauwkeuriger.
        self.C_tank = tank_liters * 0.001163  # kWh/K (soortelijke warmte water)
        self.R = 15.0  # K/kW
        self.C = 30.0  # kWh/K
        self.K_emit = 0.15  # kW/K (Afgifte vloer)
        self.K_tank = 0.25  # kW/K (Afgifte spiraal)
        self.K_loss_dhw = 0.01  # °C/uur verlies
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
                self.T_max_dhw = data.get("T_max_dhw", 58.0)
                self.ufh_lag_steps = data.get("ufh_lag_steps", 4)
                logger.info(
                    f"[Thermal] Geladen: Lag={self.ufh_lag_steps * 15}m, K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f} R={self.R:.1f} C={self.C:.1f} K_loss_dhw={self.K_loss_dhw:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to load SystemID: {e}")

    def train(self, df: pd.DataFrame):
        df_proc = clean_thermal_data(df).sort_values("timestamp")

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

        # Bereken thermisch vermogen (alleen voor rijen met echte delta_t)
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
        logger.info("[Thermal] Poging 1: Fysisch gesplitste R/C training.")

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
                    f"[Thermal] Afkoelfase geïdentificeerd met {len(df_cool)} datapunten. Geschatte Tau (RC) = {tau:.1f} uur."
                )

                # Sanity Check: Tau moet binnen realistische grenzen vallen
                if not (20 < tau < 1000):
                    logger.warning(
                        f"[Thermal] Tau waarde ({tau:.1f}) is onrealistisch. Fallback naar gecombineerde methode."
                    )
                    tau_inv = None  # Reset om de fallback te triggeren
            else:
                logger.warning(
                    "[Thermal] Negatieve coëfficiënt gevonden in afkoelfase, data is waarschijnlijk te ruizig."
                )

        else:
            logger.warning(
                f"[Thermal] Te weinig data voor afkoelfase ({len(df_cool)} punten). Fallback naar gecombineerde methode."
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
                            f"[Thermal] SUCCES (Gesplitste Methode): R={self.R:.2f}, C={self.C:.2f}"
                        )
                        # Sla model op en stop de training hier
                        # (de rest van K_emit etc. volgt na de if/else)
                    else:
                        logger.warning(
                            f"[Thermal] Gesplitste methode gaf onrealistische waarden (R={new_R:.1f}, C={new_C:.1f}). Fallback."
                        )
                        tau_inv = None  # Forceer fallback
                else:
                    logger.warning(
                        "[Thermal] Negatieve coëfficiënt gevonden in verwarmingsfase. Fallback."
                    )
                    tau_inv = None  # Forceer fallback
            else:
                logger.warning(
                    "[Thermal] Niet genoeg data voor verwarmingsfase. Fallback."
                )
                tau_inv = None  # Forceer fallback

        # ==========================================================
        # STRATEGIE 2: Gecombineerde Meervoudige Regressie (Fallback)
        # ==========================================================
        if tau_inv is None:  # Als de vorige methode niet slaagde
            logger.info(
                "[Thermal] Poging 2: Fallback naar gecombineerde multivariate regressie."
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
                        f"[Thermal] SUCCES (Fallback Methode): R={self.R:.2f}, C={self.C:.2f}"
                    )
                else:
                    logger.error(
                        "[Thermal] Beide trainingsmethodes mislukt. Coëfficiënten onlogisch. Defaults worden behouden."
                    )
            else:
                logger.error(
                    "[Thermal] Beide trainingsmethodes mislukt door te weinig data. Defaults worden behouden."
                )

        # Gebruik alleen volledige UFH kwartieren voor de lag-berekening.
        # Deelkwartieren hebben een te laag wp_output wat het gecorreleerde signaal
        # verzwakt en de gevonden lag systematisch te lang maakt.
        df_15m_lag = df_proc[
            ["timestamp", "room_temp", "wp_output", "full_quarter", "hvac_mode"]
        ].copy()
        df_15m_lag.loc[
            (df_15m_lag["hvac_mode"] == HvacMode.HEATING.value)
            & (~df_15m_lag["full_quarter"]),
            "wp_output",
        ] = np.nan  # Deelkwartieren op NaN zodat ze niet meetellen in de correlatie

        df_15m = (
            df_15m_lag[["timestamp", "room_temp", "wp_output"]]
            .set_index("timestamp")
            .resample("15min")
            .mean()
            .dropna(subset=["room_temp"])
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
                f"[Thermal] Geen duidelijke vloertraagheid gevonden (max corr={best_corr:.2f}). Fallback naar 1 uur."
            )
            self.ufh_lag_steps = 4
        else:
            self.ufh_lag_steps = int(
                np.clip(best_lag, 2, 16)
            )  # Minimaal 30 min, max 4 uur
            logger.info(
                f"[Thermal] Vloertraagheid gedetecteerd: {self.ufh_lag_steps * 15} minuten (Corr={best_corr:.2f})"
            )

        # --- LEER K_emit & K_tank (Als fallback of validatie) ---
        # Gebruik alleen volledige kwartieren: deelkwartieren hebben een te lage gemiddelde
        # delta_T (compressor warmt nog op of is al gestopt) waardoor K_emit onderschat wordt.
        mask_ufh = (
            (df_proc["hvac_mode"] == HvacMode.HEATING.value)
            & (df_proc["full_quarter"])
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
        # Gebruik alleen volledige kwartieren met return_temp beschikbaar.
        # dhw_bottom is de WP-kant van de gelaagde tank en daarmee de juiste
        # referentietemperatuur voor de warmteoverdracht van de spiraal.
        mask_dhw = (
            (df_proc["hvac_mode"] == HvacMode.DHW.value)
            & (df_proc["full_quarter"])
            & (df_proc["wp_output"] > 0.8)
            & (df_proc["supply_temp"] > df_proc["dhw_bottom"] + 1)
            & (df_proc["return_temp"].notna())
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

        # We kijken naar DHW runs en berekenen: T_max_dhw
        df_dhw = df_proc[df_proc["hvac_mode"] == HvacMode.DHW.value].copy()
        if len(df_dhw) > 10:
            # Wat is de hoogste supply_temp die we ooit in DHW mode hebben gezien?
            self.T_max_dhw = float(df_dhw["supply_temp"].max())
            logger.info(f"[Thermal] Geleerde Max DHW Temp: {self.T_max_dhw:.1f}C")

        logger.info(
            f"[Thermal] Final Trained SysID: R={self.R:.1f}, C={self.C:.1f}, "
            f"K_emit={self.K_emit:.3f}, K_tank={self.K_tank:.3f}, K_loss={self.K_loss_dhw:.3f} C_tank={self.C_tank:.3f} T_max_dhw={self.T_max_dhw:.1f}"
        )
        joblib.dump(
            {
                "R": self.R,
                "C": self.C,
                "K_emit": self.K_emit,
                "K_tank": self.K_tank,
                "K_loss_dhw": self.K_loss_dhw,
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
                    f"[Thermal] Model geladen. Geleerde parameters: UFH Factor={self.learned_factor_ufh:.2f}, DHW Factor={self.learned_factor_dhw:.2f}, UFH Lift={self.learned_lift_ufh:.2f}C, UFH Slope={self.learned_ufh_slope:.2f}, DHW Lift={self.learned_lift_dhw:.2f}C, DHW Slope={self.dhw_delta_slope:.3f} Base={self.dhw_delta_base:.1f}"
                )
            except Exception as e:
                logger.warning(f"[Thermal] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = clean_thermal_data(df)

        # 1. Bereken fysieke Delta T en check sensoren
        # Na clean_thermal_data zijn corrupte delta_t rijen al verwijderd;
        # drop hier rijen die delta_t nog missen (standby zonder meting).
        df = df.dropna(subset=["delta_t", "supply_temp", "return_temp"])
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
                f"[Thermal] UFH Geleerd: Factor={self.learned_factor_ufh:.2f}, Slope={self.learned_ufh_slope:.2f}"
            )

            # =========================================================================
            # DHW TRAINING (ROBUUST & SIMPEL)
            # =========================================================================
            # Gebruik alleen volledige kwartieren: bij deelkwartieren is supply_temp
            # nog aan het oplopen (opstartmoment) wat de lift en delta overschat.
            mask_dhw = (
                (df["hvac_mode"] == HvacMode.DHW.value)
                & (df["full_quarter"])
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
                        f"[Thermal] DHW curve Linear: Slope={power_slope:.2f}, Base={power_base:.1f}"
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
                        "[Thermal] Geëxtrapoleerde curve duikt te laag bij 35C. Base heuristisch verschoven."
                    )
                    self.dhw_delta_base += 1.5 - min_delta_at_35C

                logger.info(
                    f"[Thermal] Empirisch DHW capaciteitsmodel geaccepteerd: PowerSlope={power_slope:.2f} -> DeltaSlope={self.dhw_delta_slope:.3f}"
                )

                # Train ML Model voor Supply Temp (dit model leert ook evt niet-lineair gedrag)
                self.model_supply_dhw = RandomForestRegressor(
                    n_estimators=50, max_depth=6
                ).fit(df_dhw[self.features], df_dhw["supply_temp"])

                logger.info(
                    f"[Thermal] DHW Final: Lift={self.learned_lift_dhw:.1f}C, Base DeltaT={self.dhw_delta_base:.1f}C"
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
        val = None  # Initialiseer zodat logger.debug niet crasht als ML model ontbreekt

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
            f"[Thermal] Predict Supply: Mode={mode} P_th={p_th:.2f} T_out={t_out:.1f} T_sink={t_sink:.1f} => Pred={prediction:.1f} (Phys={physical_guess:.1f}, MinHard={min_supply_hard:.1f}) Val={val}"
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
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                self.is_fitted = True
                logger.info("[Thermal] UFH Model geladen.")
            except Exception as e:
                logger.warning(f"[Thermal] Model laden mislukt: {e}")

    def train(self, df):
        df_proc = clean_thermal_data(df).sort_values("timestamp")

        # Bereken wp_output: gebruik delta_t waar beschikbaar, anders 0 (geen schatting)
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

        # Residu berekenen (K/u)
        target = (t_next - t_model_next) / dt

        # Ruis wegpoetsen
        target = np.where(np.abs(target) < 0.15, 0, target)

        df_feat = add_cyclic_time_features(df_feat, "timestamp")
        df_feat["solar"] = df_feat["pv_actual"]
        df_feat["target"] = target
        df_feat["effective_solar"] = df_feat["solar"] * (
            df_feat.get("shutter_room", 100) / 100.0
        )

        train_set = df_feat[self.features + ["target"]].dropna()

        # Vangt extreme situaties op zoals open ramen of directe zon op de sensor
        train_set = train_set[train_set["target"].between(-1.2, 1.2)]

        if len(train_set) > 10:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=15
            ).fit(train_set[self.features], train_set["target"])
            self.is_fitted = True
            joblib.dump(self.model, self.path)
            logger.info("[Thermal] UFH model getraind.")

    def predict(self, forecast_df, shutters):
        if self.model is None or not self.is_fitted:
            logger.info("[Thermal] Geen UFH model beschikbaar, voorspel 0 residu.")
            return np.zeros(len(forecast_df))

        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        df["solar"] = df.get("power_corrected", df.get("pv_estimate", 0.0))

        # We gebruiken de AI-voorspelling van jouw eigen gedrag!
        df["shutter_room"] = shutters

        # Bereken effectieve zon
        df["effective_solar"] = df["solar"] * (df["shutter_room"] / 100.0)

        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0

        return self.model.predict(df[self.features])


class DhwResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.is_fitted = False
        self.features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                self.is_fitted = True
                logger.info("[Thermal] DHW Model geladen.")
            except Exception as e:
                logger.warning(f"[Thermal] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df = df.copy().sort_values("timestamp")

        # 1. Detecteer 'taps' (douche beurten)
        df["dhw_diff"] = df["dhw_top"].diff()

        # Filter: Alleen dalingen (>0.5 graad) terwijl de HP NIET in DHW mode staat
        mask_shower = (df["hvac_mode"] != HvacMode.DHW.value) & (df["dhw_diff"] < -0.5)

        # Maak target variabele
        df["demand"] = 0.0
        df.loc[mask_shower, "demand"] = df["dhw_diff"].abs()

        # 2. Features maken
        df = add_cyclic_time_features(df, "timestamp")

        # Selecteer data voor training
        # We trainen op ALLES, zodat het model leert wanneer het 0 is (rust) én wanneer het hoog is (douchen)
        X = df[self.features]
        y = df["demand"]

        if len(df) > 100:
            # Random Forest Regressor
            # min_samples_leaf=5 zorgt dat hij niet op 1 uniek incident traint (voorkomt overfitting)
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42
            ).fit(X, y)

            self.is_fitted = True
            joblib.dump(self.model, self.path)
            logger.info("[Thermal] DHW model getraind.")
        else:
            logger.warning("[Shower] Te weinig data om te trainen.")

    def predict(self, forecast_df):
        """Geeft een array terug met de verwachte daling (graden) per tijdstap."""
        if self.model is None or not self.is_fitted:
            logger.info("[Thermal] Geen DHW model beschikbaar, voorspel 0 daling.")
            return np.zeros(len(forecast_df))

        # Maak features voor de forecast
        df_feat = add_cyclic_time_features(forecast_df.copy(), "timestamp")

        # Voorspel de daling
        predictions = self.model.predict(df_feat[self.features])

        # 3. Post-processing (Belangrijk voor MPC!)
        # Een Random Forest smeert vaak uit: hij voorspelt 0.5 graden over 4 kwartieren
        # in plaats van 2.0 graden in 1 kwartier.
        # Voor de optimizer maakt dit niet heel veel uit (de integraal is hetzelfde),
        # maar we willen ruis (0.1 graad continu) negeren.

        # Filter: negeer alles onder de 0.8 graad daling per kwartier
        predictions = np.where(predictions < 0.8, 0.0, predictions)

        # Boost de pieken iets om zeker te zijn dat de buffer groot genoeg is
        predictions = predictions * 1.5

        return predictions
