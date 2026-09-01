import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.optimize import least_squares

from domain.models import (
    BoilerThermalModel,
    Config,
    DatasetDefinition,
    Forecaster,
    ForecasterType,
)
from features.dataset import DatasetBuilder

logger = logging.getLogger(__name__)


class GreyBoxForecaster(Forecaster):
    def __init__(self):
        self.params_: dict[str, float] = {}
        self.metadata_: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.params_)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        dump(
            {"params": self.params_, "metadata": self.metadata_},
            path / f"{self.name}.joblib",
        )
        logger.info(
            "[%s] Model opgeslagen naar %s", self.name, path / f"{self.name}.joblib"
        )

    def load(self, path: Path, study_storage: str | None = None) -> None:
        file_path = path / f"{self.name}.joblib"
        if not file_path.exists():
            return
        data = load(file_path)
        if isinstance(data, dict):
            self.params_ = data.get("params", {})
            self.metadata_ = data.get("metadata", {})

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def to_thermal_model(self) -> Any: ...


class BoilerForecaster(GreyBoxForecaster):
    """
    100% Zelflerend Fysisch Boiler Model (Multiple Shooting / Sub-Trajectory Identificatie).
    Vrij van arbitraire tapwater-drempels of heuristieken.
    """

    CP = 4.186  # kJ/kg·K (Constante van water)
    RHO = 1.0  # kg/L
    DT_HOURS = 0.25  # 15 minuten

    # Horizonten voor traject-integratie (Multiple Shooting)
    IDLE_HORIZON_STEPS = 12  # 3 uur stilstand (voldoende om 1-stapsruis te elimineren)
    SWW_HORIZON_STEPS = 4  # 1 uur SWW opwarming

    DEFAULT_PARAMS = {
        "C_top": 0.1163,
        "C_bottom": 0.1163,
        "UA_top": 0.0014,
        "UA_bottom": 0.0014,
        "k_cross": 0.0005,
        "f_top": 0.50,
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

        delta_t = (df["T_aanvoer"] - df["T_retour"]).clip(lower=0.0)
        mass_flow = (df["flow_lpm"].clip(lower=0.0) / 60.0) * self.RHO
        df["Q_hp"] = mass_flow * self.CP * delta_t

        clean = df.dropna(subset=["T_top", "T_bottom", "Q_hp", "T_ambient"]).copy()
        if clean.empty:
            raise ValueError("Geen geldige trainingsdata overgebleven.")

        return clean

    def _extract_subtrajectories(self, df: pd.DataFrame) -> list[dict[str, np.ndarray]]:
        """
        Deelt de tijdreeks op in NIET-OVERLAPPENDE zuivere trajecten van 3 uur.
        Hierdoor telt een douchebeurt slechts 1x mee en wordt hij door Cauchy loss verworpen.
        """
        is_sww = (df["state"].astype(str).str.upper() == "SWW") & (df["Q_hp"] > 1.0)
        is_idle = (df["Q_hp"] < 0.1) & (~is_sww)

        regime = np.where(is_sww, 1, np.where(is_idle, 0, -1))
        regime_changes = regime != np.roll(regime, 1)
        regime_changes[0] = True
        block_ids = np.cumsum(regime_changes)

        trajectories = []
        for _, group in df.groupby(block_ids):
            is_sww_block = is_sww.loc[group.index].all()
            is_idle_block = is_idle.loc[group.index].all()
            n = len(group)

            if is_idle_block and n >= self.IDLE_HORIZON_STEPS:
                # NIET-OVERLAPPEND: stap per 12 kwartieren (exact 3 uur per venster)
                for start in range(
                    0, n - self.IDLE_HORIZON_STEPS + 1, self.IDLE_HORIZON_STEPS
                ):
                    sub = group.iloc[start : start + self.IDLE_HORIZON_STEPS]
                    trajectories.append(
                        {
                            "t_top": sub["T_top"].to_numpy(dtype=float),
                            "t_bot": sub["T_bottom"].to_numpy(dtype=float),
                            "t_amb": sub["T_ambient"].to_numpy(dtype=float),
                            "q_hp": sub["Q_hp"].to_numpy(dtype=float),
                            "n_steps": len(sub),
                        }
                    )

            elif is_sww_block and n >= 2:
                # SWW opwarmtraject (minstens 30 min)
                trajectories.append(
                    {
                        "t_top": group["T_top"].to_numpy(dtype=float),
                        "t_bot": group["T_bottom"].to_numpy(dtype=float),
                        "t_amb": group["T_ambient"].to_numpy(dtype=float),
                        "q_hp": group["Q_hp"].to_numpy(dtype=float),
                        "n_steps": len(group),
                    }
                )

        return trajectories

    def _simulate_and_evaluate(
        self, theta: np.ndarray, trajectories: list[dict[str, np.ndarray]]
    ) -> np.ndarray:
        c_layer, ua_layer, k_cross, f_top = theta
        c_top = c_layer
        c_bot = c_layer
        ua_top = ua_layer
        ua_bot = ua_layer
        dt = self.DT_HOURS
        residuals = []

        for traj in trajectories:
            n = traj["n_steps"]
            sim_top = np.empty(n)
            sim_bot = np.empty(n)

            # Startconditie = beginwaarde van dit sub-traject
            sim_top[0] = traj["t_top"][0]
            sim_bot[0] = traj["t_bot"][0]

            t_amb = traj["t_amb"]
            q_hp = traj["q_hp"]

            for t in range(n - 1):
                T_t = sim_top[t]
                T_b = sim_bot[t]

                dT_top = (
                    f_top * q_hp[t] + ua_top * (t_amb[t] - T_t) + k_cross * (T_b - T_t)
                ) / c_top
                dT_bot = (
                    (1.0 - f_top) * q_hp[t]
                    + ua_bot * (t_amb[t] - T_b)
                    - k_cross * (T_b - T_t)
                ) / c_bot

                sim_top[t + 1] = T_t + dT_top * dt
                sim_bot[t + 1] = T_b + dT_bot * dt

            residuals.append(sim_top - traj["t_top"])
            residuals.append(sim_bot - traj["t_bot"])

        return np.concatenate(residuals)

    def fit(self, df: pd.DataFrame) -> None:
        df_clean = self.prepare(df)
        trajectories = self._extract_subtrajectories(df_clean)

        if not trajectories:
            raise ValueError(
                "Onvoldoende data om dynamische trajecten uit te extraheren."
            )

        # Startwaarden: [C_layer, UA_layer, k_cross, f_top]
        x0 = np.array([0.1163, 0.0010, 0.0004, 0.50])

        # Strikte fysische grenzen:
        # - f_top tussen 0.48 en 0.55 (warmte stijgt op -> top >= bodem)
        # - k_cross minimaal 0.2 W/K (waterkolom geleidt altijd warmte)
        bounds = (
            [0.0900, 0.0005, 0.00020, 0.48],  # lower bounds
            [0.1400, 0.0030, 0.00200, 0.55],  # upper bounds
        )

        # Cauchy Robuuste Loss met f_scale=0.15:
        # Elimineert uitschieters (douchen) wiskundig veel agressiever dan Soft-L1
        result = least_squares(
            fun=self._simulate_and_evaluate,
            x0=x0,
            args=(trajectories,),
            bounds=bounds,
            method="trf",
            loss="cauchy",
            f_scale=0.15,
            max_nfev=600,
        )

        if not result.success:
            raise RuntimeError(f"Systeemidentificatie mislukt: {result.message}")

        c_fit, ua_fit, k_fit, f_fit = result.x

        self.params_ = {
            "C_top": float(c_fit),
            "C_bottom": float(c_fit),
            "UA_top": float(ua_fit),
            "UA_bottom": float(ua_fit),
            "k_cross": float(k_fit),
            "f_top": float(f_fit),
        }

        is_sww = (df_clean["state"].astype(str).str.upper() == "SWW") & (
            df_clean["Q_hp"] > 1.0
        )
        sww_data = df_clean[is_sww]
        typical_q_hp = (
            float(sww_data["Q_hp"].median())
            if not sww_data.empty
            else self.DEFAULT_TYPICAL_Q_HP
        )

        res = self._simulate_and_evaluate(result.x, trajectories)
        rmse = float(np.sqrt(np.mean(res**2)))

        self.metadata_ = {
            "typical_q_hp": typical_q_hp,
            "rmse": rmse,
            "trajectories_count": len(trajectories),
        }

        total_volume_l = (
            (self.params_["C_top"] + self.params_["C_bottom"])
            * 3600.0
            / (self.RHO * self.CP)
        )
        ua_total_w_k = (self.params_["UA_top"] + self.params_["UA_bottom"]) * 1000.0

        print("=== RESULTAAT ZUIVERE NAUWKEURIGE SYSTEEMIDENTIFICATIE ===")
        print(
            f"[GreyBox] Geanalyseerde trajecten: {len(trajectories)} (RMSE = {rmse:.3f} °C)"
        )
        print(
            f"[GreyBox] Geleerd volume:          {total_volume_l:.1f} Liter (C_layer={c_fit:.4f} kWh/°C)"
        )
        print(
            f"[GreyBox] Geleerd isolatieverlies:   UA_totaal = {ua_total_w_k:.2f} W/K ({ua_fit * 1000:.2f} W/K per zone)"
        )
        print(f"[GreyBox] Geleerde geleiding:       k_cross = {k_fit * 1000:.2f} W/K")
        print(
            f"[GreyBox] Geleerde warmtestroom:    f_top = {f_fit * 100:.1f}% naar de top"
        )
        print(f"[GreyBox] Typisch vermogen:         Q_hp = {typical_q_hp:.2f} kW")

    def to_thermal_model(self) -> BoilerThermalModel:
        p = self.params_ if self.is_fitted else self.DEFAULT_PARAMS
        typical_q_hp = (
            self.metadata_.get("typical_q_hp", self.DEFAULT_TYPICAL_Q_HP)
            if self.is_fitted
            else self.DEFAULT_TYPICAL_Q_HP
        )
        dt = self.DT_HOURS

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
