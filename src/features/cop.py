import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from domain.types import (
    Config,
    HeatPumpCOPModel,
)
from features.dataset import DatasetBuilder, DatasetDefinition
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class HeatPumpCOPIdentifier(SystemIdentifier[HeatPumpCOPModel]):
    TRAIN_RATIO = 0.80

    MIN_FLOW_LPM = 0.0
    MIN_ELECTRICAL_POWER_KW = 0.1

    MIN_COP = 1.0
    MAX_COP = 10.0

    MIN_ETA = 0.05
    MAX_ETA = 0.90

    MIN_DELTA_T_COND = 0.5
    MAX_DELTA_T_COND = 20.0

    MIN_DELTA_T_EVAP = 0.5
    MAX_DELTA_T_EVAP = 20.0

    INITIAL_ETA = 0.45
    INITIAL_DELTA_T_COND = 5.0
    INITIAL_DELTA_T_EVAP = 5.0

    @property
    def name(self) -> str:
        return "heat_pump_cop"

    @property
    def label(self) -> str:
        return "COP"

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        required_columns = [
            "T_ambient",
            "T_supply",
            "T_return",
            "T_setpoint",
            "flow_lpm",
            "P_el",
            "state",
        ]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        numeric_columns = [
            "T_ambient",
            "T_supply",
            "T_return",
            "T_setpoint",
            "flow_lpm",
            "P_el",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(subset=numeric_columns).copy()

        df["delta_t_water"] = df["T_supply"] - df["T_return"]

        df["Q_th"] = 0.06978 * df["flow_lpm"] * df["delta_t_water"]

        df["COP_measured"] = df["Q_th"] / df["P_el"]

        valid = (
            (df["flow_lpm"] > self.MIN_FLOW_LPM)
            & (df["P_el"] > self.MIN_ELECTRICAL_POWER_KW)
            & (df["delta_t_water"] > 0.0)
            & (df["Q_th"] > 0.0)
            & (df["COP_measured"] > self.MIN_COP)
            & (df["COP_measured"] < self.MAX_COP)
        )

        invalid_count = int((~valid).sum())

        df = df.loc[valid].copy()

        logger.info(
            "Heat pump COP preparation: %d valid points, %d points removed",
            len(df),
            invalid_count,
        )

        if df.empty:
            raise ValueError(
                "No valid heat-pump COP measurements remain after filtering."
            )

        return df

    @staticmethod
    def _predict_cop(
        parameters: np.ndarray,
        T_ambient: np.ndarray,
        T_supply: np.ndarray,
    ) -> np.ndarray:

        eta_carnot, delta_t_cond, delta_t_evap = parameters

        T_cond_C = T_supply + delta_t_cond

        T_evap_C = T_ambient - delta_t_evap

        T_cond_K = T_cond_C + 273.15

        T_evap_K = T_evap_C + 273.15

        temperature_lift = T_cond_K - T_evap_K

        return eta_carnot * T_cond_K / temperature_lift

    def calibrate(
        self,
        df: pd.DataFrame,
    ) -> HeatPumpCOPModel:

        if len(df) < 10:
            raise ValueError("Not enough data points for calibration.")

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 0 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].copy()

        logger.info(
            "Heat pump COP calibration: %d training points, %d validation points",
            len(train_df),
            len(df) - len(train_df),
        )

        T_ambient = train_df["T_ambient"].to_numpy(dtype=float)

        T_supply = train_df["T_supply"].to_numpy(dtype=float)

        COP_measured = train_df["COP_measured"].to_numpy(dtype=float)

        def residuals(
            parameters: np.ndarray,
        ) -> np.ndarray:

            COP_predicted = self._predict_cop(
                parameters,
                T_ambient,
                T_supply,
            )

            return COP_predicted - COP_measured

        x0 = np.array(
            [
                self.INITIAL_ETA,
                self.INITIAL_DELTA_T_COND,
                self.INITIAL_DELTA_T_EVAP,
            ]
        )

        lower_bounds = np.array(
            [
                self.MIN_ETA,
                self.MIN_DELTA_T_COND,
                self.MIN_DELTA_T_EVAP,
            ]
        )

        upper_bounds = np.array(
            [
                self.MAX_ETA,
                self.MAX_DELTA_T_COND,
                self.MAX_DELTA_T_EVAP,
            ]
        )

        result = least_squares(
            residuals,
            x0=x0,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
        )

        if not result.success:
            logger.warning(
                "Heat pump COP calibration did not fully converge: %s",
                result.message,
            )

        eta_carnot = float(result.x[0])
        delta_t_cond = float(result.x[1])
        delta_t_evap = float(result.x[2])

        logger.info(
            "Heat pump COP parameters calibrated: "
            "eta_carnot=%.4f, "
            "delta_t_cond=%.3f K, "
            "delta_t_evap=%.3f K",
            eta_carnot,
            delta_t_cond,
            delta_t_evap,
        )

        self.model = HeatPumpCOPModel(
            eta_carnot=eta_carnot,
            delta_t_cond=delta_t_cond,
            delta_t_evap=delta_t_evap,
        )

        return self.model

    def validate(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float]:

        if self.model is None:
            raise RuntimeError("Model must be calibrated before validation.")

        if len(df) < 2:
            raise ValueError("Not enough data points for validation.")

        split_index = int(len(df) * self.TRAIN_RATIO)

        test_df = df.iloc[split_index:].copy()

        if test_df.empty:
            raise ValueError("No validation data available.")

        T_ambient = test_df["T_ambient"].to_numpy(dtype=float)

        T_supply = test_df["T_supply"].to_numpy(dtype=float)

        COP_measured = test_df["COP_measured"].to_numpy(dtype=float)

        parameters = np.array(
            [
                self.model.eta_carnot,
                self.model.delta_t_cond,
                self.model.delta_t_evap,
            ]
        )

        COP_predicted = self._predict_cop(
            parameters,
            T_ambient,
            T_supply,
        )

        r2 = float(
            r2_score(
                COP_measured,
                COP_predicted,
            )
        )

        mae = float(
            mean_absolute_error(
                COP_measured,
                COP_predicted,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    COP_measured,
                    COP_predicted,
                )
            )
        )

        logger.info(
            "Heat pump COP validation: R2=%.4f, MAE=%.4f, RMSE=%.4f",
            r2,
            mae,
            rmse,
        )

        logger.info(
            "Heat pump COP validation: measured mean=%.3f, predicted mean=%.3f",
            float(np.mean(COP_measured)),
            float(np.mean(COP_predicted)),
        )

        return {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }

    def dataset(
        self,
        config: Config,
    ) -> DatasetDefinition:

        return (
            DatasetBuilder()
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_supply",
                config.heat_pump.supply_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_return",
                config.heat_pump.return_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "flow_lpm",
                config.heat_pump.flow,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "P_el",
                config.heat_pump.power,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="5m",
                aggregation="last",
                fill="previous",
            )
            .build()
        )
