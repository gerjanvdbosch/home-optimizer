import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.optimize import least_squares

from domain.models import (
    BacktestPoint,
    BacktestResult,
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

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.DataFrame: ...

    @abstractmethod
    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult: ...

    @abstractmethod
    def to_thermal_model(self) -> BoilerThermalModel: ...

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
            if self.params_:
                self._update_model_from_params()

    @abstractmethod
    def _update_model_from_params(self) -> None: ...


class BoilerForecaster(GreyBoxForecaster):
    """100% Zelflerend Fysisch Boiler Model (Multiple Shooting / Sub-Trajectory Identificatie).

    Biedt volledige interoperabiliteit met predict() en backtest().
    """

    CP = 4.186  # kJ/kg·K (Constante water)
    RHO = 1.0  # kg/L
    DT_HOURS = 0.25  # 15 minuten

    IDLE_HORIZON_STEPS = 12  # 3 uur stilstand per sub-traject
    SWW_HORIZON_STEPS = 4  # 1 uur opwarming

    def __init__(self) -> None:
        super().__init__()
        self._model = BoilerThermalModel(dt_hours=self.DT_HOURS)

    @property
    def name(self) -> ForecasterType:
        return "boiler"

    @property
    def label(self) -> str:
        return "Boiler Temperature"

    @property
    def unit(self) -> str:
        return "°C"

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
        c_layer, ua_layer, k_idle, k_mix, f_top = theta
        temp_model = BoilerThermalModel(
            c_top=c_layer,
            c_bottom=c_layer,
            ua_top=ua_layer,
            ua_bottom=ua_layer,
            k_idle=k_idle,
            k_mix=k_mix,
            f_top=f_top,
            dt_hours=self.DT_HOURS,
        )

        residuals = []
        for traj in trajectories:
            sim_top, sim_bot = temp_model.simulate(
                initial_top=traj["t_top"][0],
                initial_bottom=traj["t_bot"][0],
                ambient_temps=traj["t_amb"],
                q_hp_profile=traj["q_hp"],
            )
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

        # Startwaarden: [C_layer, UA_layer, k_idle, k_mix, f_top]
        x0 = np.array([0.1163, 0.0010, 0.0003, 0.0400, 0.50])

        bounds = (
            [0.0800, 0.0005, 0.00010, 0.0050, 0.50],  # f_top >= 0.50 afgedwongen
            [0.1500, 0.0030, 0.00150, 0.2000, 0.65],  # k_mix tot 0.200 W/K toegestaan
        )

        result = least_squares(
            fun=self._simulate_and_evaluate,
            x0=x0,
            args=(trajectories,),
            bounds=bounds,
            method="trf",
            loss="cauchy",
            f_scale=0.15,
            max_nfev=800,
        )

        if not result.success:
            raise RuntimeError(f"Systeemidentificatie mislukt: {result.message}")

        c_fit, ua_fit, k_idle_fit, k_mix_fit, f_fit = result.x

        is_sww = (df_clean["state"].astype(str).str.upper() == "SWW") & (
            df_clean["Q_hp"] > 1.0
        )
        sww_data = df_clean[is_sww]
        typical_q_hp = (
            float(sww_data["Q_hp"].median())
            if not sww_data.empty
            else BoilerThermalModel.typical_q_hp_kw
        )

        self.params_ = {
            "c_top": float(c_fit),
            "c_bottom": float(c_fit),
            "ua_top": float(ua_fit),
            "ua_bottom": float(ua_fit),
            "k_idle": float(k_idle_fit),
            "k_mix": float(k_mix_fit),
            "f_top": float(f_fit),
            "typical_q_hp_kw": float(typical_q_hp),
        }

        res = self._simulate_and_evaluate(result.x, trajectories)
        rmse = float(np.sqrt(np.mean(res**2)))

        # Berekening van afgeleide fysische grootheden
        total_volume_l = (
            (self.params_["c_top"] + self.params_["c_bottom"])
            * 3600.0
            / (self.RHO * self.CP)
        )
        ua_total_w_k = (self.params_["ua_top"] + self.params_["ua_bottom"]) * 1000.0
        k_idle_w_k = self.params_["k_idle"] * 1000.0
        k_mix_w_k = self.params_["k_mix"] * 1000.0

        # Sla alle nuttige metadata op
        self.metadata_ = {
            "typical_q_hp": typical_q_hp,
            "rmse": rmse,
            "trajectories_count": len(trajectories),
            "total_volume_liters": round(total_volume_l, 1),
            "ua_total_w_k": round(ua_total_w_k, 2),
            "k_idle_w_k": round(k_idle_w_k, 2),
            "k_mix_w_k": round(k_mix_w_k, 2),
        }

        # Model instance synchroniseren
        self._update_model_from_params()

        # Print / Log samenvatting
        summary = (
            f"\n=== RESULTAAT ZUIVERE NAUWKEURIGE SYSTEEMIDENTIFICATIE ===\n"
            f"[GreyBox] Geanalyseerde trajecten: {len(trajectories)} (RMSE = {rmse:.3f} °C)\n"
            f"[GreyBox] Geleerd volume:          {total_volume_l:.1f} Liter (C_layer = {c_fit:.4f} kWh/°C)\n"
            f"[GreyBox] Geleerd isolatieverlies:   UA_totaal = {ua_total_w_k:.2f} W/K ({ua_fit * 1000:.2f} W/K per zone)\n"
            f"[GreyBox] Geleerde rustgeleiding:    k_idle = {k_idle_w_k:.2f} W/K\n"
            f"[GreyBox] Geleerde actieve menging:  k_mix = {k_mix_w_k:.2f} W/K\n"
            f"[GreyBox] Geleerde warmtestroom:     f_top = {f_fit * 100:.1f}% naar de top\n"
            f"[GreyBox] Typisch vermogen:          Q_hp = {typical_q_hp:.2f} kW\n"
            f"==========================================================="
        )
        logger.info(summary)

    def _update_model_from_params(self) -> None:
        self._model = BoilerThermalModel(
            c_top=self.params_["c_top"],
            c_bottom=self.params_["c_bottom"],
            ua_top=self.params_["ua_top"],
            ua_bottom=self.params_["ua_bottom"],
            k_idle=self.params_["k_idle"],
            k_mix=self.params_["k_mix"],
            f_top=self.params_["f_top"],
            typical_q_hp_kw=self.params_.get("typical_q_hp_kw", 5.5),
            dt_hours=self.DT_HOURS,
        )

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.DataFrame:
        """Voert een voorwaartse simulatie uit o.b.v. een gepland aanstuurprofiel in df."""
        df_clean = self.prepare(df).iloc[:steps]

        sim_top, sim_bot = self._model.simulate(
            initial_top=float(df_clean["T_top"].iloc[0]),
            initial_bottom=float(df_clean["T_bottom"].iloc[0]),
            ambient_temps=df_clean["T_ambient"].to_numpy(dtype=float),
            q_hp_profile=df_clean["Q_hp"].to_numpy(dtype=float),
        )

        return pd.DataFrame(
            {"pred_top": sim_top, "pred_bottom": sim_bot},
            index=df_clean.index,
        )

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        """Valideert het geïdentificeerde model over een test dataset."""
        df_clean = self.prepare(df)
        preds = self.predict(df_clean, steps=len(df_clean))

        mae = float(np.mean(np.abs(preds["pred_top"] - df_clean["T_top"])))

        def make_points(label: str, series: pd.Series) -> BacktestPoint:
            pts = (
                series.rename("value")
                .rename_axis("time")
                .reset_index()
                .to_dict("records")
            )
            return BacktestPoint(label=label, points=pts)

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=mae,
            points=[
                make_points("Actual Top", df_clean["T_top"]),
                make_points("Predicted Top", preds["pred_top"]),
            ],
        )

    def to_thermal_model(self) -> BoilerThermalModel:
        return self._model

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
