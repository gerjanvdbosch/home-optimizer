import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from optuna import Trial
from scipy.optimize import least_squares
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from domain.models import (
    BoilerThermalModel,
    Config,
    DatasetDefinition,
    Forecaster,
    ForecasterType,
)
from features.dataset import DatasetBuilder
from features.forecaster import SkforecastForecaster, SklearnForecaster

logger = logging.getLogger(__name__)


class GreyBoxForecaster(Forecaster):
    """
    Base class voor fysische / grey-box modellen (zoals boilers, RC-schillen van huizen).
    Beheert het opslaan, inladen en valideren van geschatte fysische parameters.
    """

    def __init__(self):
        self.params_: dict[str, float] = {}
        self.metadata_: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """Geeft aan of het model al getraind of ingeladen is."""
        return bool(self.params_)

    def save(self, path: Path) -> None:
        """Slaat de geschatte parameters en metadata atomair op naar disk."""
        path.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "params": self.params_,
                "metadata": self.metadata_,
            },
            path / f"{self.name}.joblib",
        )
        logger.info(
            "[%s] Model opgeslagen naar %s", self.name, path / f"{self.name}.joblib"
        )

    def load(self, path: Path, study_storage: str | None = None) -> None:
        """Laadt de parameters en metadata in vanuit disk."""
        file_path = path / f"{self.name}.joblib"
        if not file_path.exists():
            logger.warning(
                "[%s] Geen opgeslagen model gevonden op %s", self.name, file_path
            )
            return

        data = load(file_path)
        if isinstance(data, dict):
            self.params_ = data.get("params", {})
            self.metadata_ = data.get("metadata", {})
            logger.info(
                "[%s] Model succesvol ingeladen. Parameters: %s",
                self.name,
                self.params_,
            )
        else:
            logger.error("[%s] Onverwacht formaat in %s", self.name, file_path)

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Schat de fysische parameters (bijv. via least_squares of Kalman filtering)."""
        ...

    @abstractmethod
    def to_thermal_model(self) -> Any:
        """Exporteert het getrainde model naar een state-space object voor MPC."""
        ...


class BoilerForecaster(GreyBoxForecaster):
    """
    Fysisch 2-laags boiler model gebaseerd op de 1D energiebeschermingswet.

    Vaste fysische constanten (200L tank):
      - C_top = C_bottom = 0.1163 kWh/°C (100L water per laag)

    Te schatten fysische parameters:
      - UA_top, UA_bottom [kW/°C] : Wandisolatieverlies
      - k_cross           [kW/°C] : Statische warmtegeleiding tussen lagen
      - f_top             [-]     : Effectieve warmteverdeling van de spiraal
    """

    CP = 4.186  # kJ/kg·K (Soortelijke warmte water)
    RHO = 1.0  # kg/L (Dichtheid water)
    DT_HOURS = 0.25  # 15 minuten tijdstap

    # Eerste-orde fysische eigenschappen (200L boiler)
    VOLUME_TOTAL_L = 200.0
    C_LAYER = (VOLUME_TOTAL_L / 2.0) * RHO * CP / 3600.0  # = 0.1163 kWh/°C

    # Standaard fysische startwaarden (ErP Label B isolatie)
    DEFAULT_PARAMS = {
        "C_top": C_LAYER,
        "C_bottom": C_LAYER,
        "UA_top": 0.0008,  # 0.8 W/K per zone (totaal 1.6 W/K = ~35W stilstandsverlies bij dT=40)
        "UA_bottom": 0.0008,  # 0.8 W/K
        "k_cross": 0.0005,  # 0.5 W/K geleiding door waterkolom
        "f_top": 0.50,  # Convectieve opstijging verdeelt warmte gelijkmatig
    }
    DEFAULT_TYPICAL_Q_HP = 5.5

    @property
    def name(self) -> ForecasterType:
        return "boiler"

    @property
    def label(self) -> str:
        return "Temperature"

    @property
    def unit(self) -> str:
        return "°C"

    @property
    def target_column(self) -> str:
        return "T_top"

    @property
    def exog_columns(self) -> list[str]:
        return ["Q_hp", "T_ambient"]

    # ------------------------------------------------------------------
    # 1. Zuivere data preparatie
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "time" in df.columns:
            df = df.set_index("time")
        df = df.sort_index().asfreq("15min")

        required = [
            "T_ambient",
            "T_top",
            "T_bottom",
            "T_aanvoer",
            "T_retour",
            "flow_lpm",
            "state",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset mist vereiste kolom(men): {missing}")

        # Thermisch vermogen: Q_hp = mass_flow * cp * delta_T (in kW)
        delta_t = (df["T_aanvoer"] - df["T_retour"]).clip(lower=0.0)
        mass_flow = (df["flow_lpm"].clip(lower=0.0) / 60.0) * self.RHO
        df["Q_hp"] = mass_flow * self.CP * delta_t

        df["T_top_next"] = df["T_top"].shift(-1)
        df["T_bottom_next"] = df["T_bottom"].shift(-1)

        # Alleen rijen zonder ontbrekende sensorwaarden
        clean = df.dropna(
            subset=[
                "T_top",
                "T_bottom",
                "T_top_next",
                "T_bottom_next",
                "Q_hp",
                "T_ambient",
            ]
        ).copy()

        if clean.empty:
            raise ValueError("Geen geldige trainingsdata overgebleven.")

        return clean

    # ------------------------------------------------------------------
    # 2. Fysische 1D Energiebalans
    # ------------------------------------------------------------------
    def _residuals(
        self, theta: np.ndarray, arrays: dict[str, np.ndarray]
    ) -> np.ndarray:
        ua_top, ua_bot, k_cross, f_top = theta
        c_top = self.C_LAYER
        c_bot = self.C_LAYER
        dt = self.DT_HOURS

        t_top = arrays["t_top"]
        t_bot = arrays["t_bot"]
        q_hp = arrays["q_hp"]
        t_amb = arrays["t_amb"]

        # Exacte 1D differentiaalvergelijkingen
        dT_top = (
            f_top * q_hp + ua_top * (t_amb - t_top) + k_cross * (t_bot - t_top)
        ) / c_top
        dT_bot = (
            (1.0 - f_top) * q_hp + ua_bot * (t_amb - t_bot) - k_cross * (t_bot - t_top)
        ) / c_bot

        pred_top = t_top + dT_top * dt
        pred_bot = t_bot + dT_bot * dt

        # Residuen
        res_top = pred_top - arrays["t_top_next"]
        res_bot = pred_bot - arrays["t_bottom_next"]

        return np.concatenate([res_top, res_bot])

    # ------------------------------------------------------------------
    # 3. Fitting via Robuuste M-Estimatie (Soft-L1)
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        df_clean = self.prepare(df)

        arrays = {
            "t_top": df_clean["T_top"].to_numpy(),
            "t_bot": df_clean["T_bottom"].to_numpy(),
            "t_amb": df_clean["T_ambient"].to_numpy(),
            "q_hp": df_clean["Q_hp"].to_numpy(),
            "t_top_next": df_clean["T_top_next"].to_numpy(),
            "t_bottom_next": df_clean["T_bottom_next"].to_numpy(),
        }

        # Parameters om te schatten: [UA_top, UA_bot, k_cross, f_top]
        x0 = np.array([0.0008, 0.0008, 0.0005, 0.50])
        bounds = (
            [
                0.0002,
                0.0002,
                0.0001,
                0.40,
            ],  # lower (fysisch minimale isolatie en f_top)
            [0.0030, 0.0030, 0.0050, 0.60],  # upper
        )

        # Soft-L1 loss met f_scale=0.15 °C:
        # Fouten < 0.15 °C (natuurlijke afkoeling/opwarming) worden kwadratisch geminimaliseerd.
        # Grote uitschieters (douchen/tappen) worden lineair gedempt en vervuilen de fit niet!
        result = least_squares(
            fun=self._residuals,
            x0=x0,
            args=(arrays,),
            bounds=bounds,
            method="trf",
            loss="soft_l1",
            f_scale=0.15,
            max_nfev=500,
        )

        if not result.success:
            raise RuntimeError(f"Fitting mislukt: {result.message}")

        self.params_ = {
            "C_top": self.C_LAYER,
            "C_bottom": self.C_LAYER,
            "UA_top": float(result.x[0]),
            "UA_bottom": float(result.x[1]),
            "k_cross": float(result.x[2]),
            "f_top": float(result.x[3]),
        }

        # Bepaal het werkelijke thermisch vermogen tijdens SWW
        is_sww = df_clean["state"].astype(str).str.strip().str.upper() == "SWW"
        sww = df_clean[is_sww & (df_clean["Q_hp"] > 1.0)]
        typical_q_hp = (
            float(sww["Q_hp"].median()) if not sww.empty else self.DEFAULT_TYPICAL_Q_HP
        )

        self.metadata_ = {
            "typical_q_hp": typical_q_hp,
            "rmse": float(np.sqrt(np.mean(self._residuals(result.x, arrays) ** 2))),
        }

        print(f"[GreyBox] Fysische kalibratie voltooid.")
        print(
            f"[GreyBox] Vaste capaciteiten: C_top = C_bot = {self.C_LAYER:.4f} kWh/°C"
        )
        print(f"[GreyBox] Geschatte parameters: {self.params_}")
        print(f"[GreyBox] Typisch vermogen warmtepomp: {typical_q_hp:.2f} kW")

    # ------------------------------------------------------------------
    # 4. State-Space Generatie voor MPC
    # ------------------------------------------------------------------
    def to_thermal_model(self) -> BoilerThermalModel:
        p = self.params_ if self.is_fitted else self.DEFAULT_PARAMS
        typical_q_hp = (
            self.metadata_.get("typical_q_hp", self.DEFAULT_TYPICAL_Q_HP)
            if self.is_fitted
            else self.DEFAULT_TYPICAL_Q_HP
        )
        dt = self.DT_HOURS

        # Discrete A, B matrices volgens de continu-naar-discreet Euler transformatie
        a_top_top = 1.0 - dt * (p["UA_top"] + p["k_cross"]) / p["C_top"]
        a_top_bottom = dt * p["k_cross"] / p["C_top"]
        c_top = dt * p["f_top"] / p["C_top"]

        a_bottom_top = dt * p["k_cross"] / p["C_bottom"]
        a_bottom_bottom = 1.0 - dt * (p["UA_bottom"] + p["k_cross"]) / p["C_bottom"]
        c_bottom = dt * (1.0 - p["f_top"]) / p["C_bottom"]

        return BoilerThermalModel(
            a_top_top=float(a_top_top),
            a_top_bottom=float(a_top_bottom),
            c_top=float(c_top),
            a_bottom_top=float(a_bottom_top),
            a_bottom_bottom=float(a_bottom_bottom),
            c_bottom=float(c_bottom),
            typical_q_hp_kw=float(typical_q_hp),
        )

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_aanvoer",
                config.heat_pump.supply_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_retour",
                config.heat_pump.return_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "flow_lpm",
                config.heat_pump.flow,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            .build()
        )
