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
    """
    Minimal semi-physical heat-pump COP identifier.

    Model:

        COP = eta_carnot * COP_carnot

        COP_carnot =
            T_cond / (T_cond - T_evap)

        T_cond = T_aanvoer + delta_t_cond
        T_evap = T_ambient - delta_t_evap

    Temperatures used in the Carnot equation are converted to Kelvin.

    Measured thermal power:

        Q_th [kW] =
            0.06978 * flow_lpm * (T_aanvoer - T_retour)

    Measured COP:

        COP_measured = Q_th / P_el
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    TRAIN_RATIO = 0.80

    # Sanity limits for identification.
    # These should eventually be made configurable for the installation.
    MIN_FLOW_LPM = 0.0
    MIN_ELECTRICAL_POWER_KW = 0.1

    MIN_COP = 1.0
    MAX_COP = 10.0

    # Parameter bounds
    MIN_ETA = 0.05
    MAX_ETA = 0.90

    MIN_DELTA_T_COND = 0.5
    MAX_DELTA_T_COND = 20.0

    MIN_DELTA_T_EVAP = 0.5
    MAX_DELTA_T_EVAP = 20.0

    # Initial parameter values
    INITIAL_ETA = 0.45
    INITIAL_DELTA_T_COND = 5.0
    INITIAL_DELTA_T_EVAP = 5.0

    # ------------------------------------------------------------------
    # Identifier metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "heat_pump_cop"

    @property
    def label(self) -> str:
        return "Heat Pump COP"

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare raw heat-pump measurements for calibration.

        Required input columns:

            T_ambient   [°C]
            T_aanvoer   [°C]
            T_retour    [°C]
            T_setpoint  [°C]
            flow_lpm    [L/min]
            P_el        [kW]
            state       [-]

        Adds:

            delta_t_water
            Q_th
            COP_measured
        """

        df = df.copy()

        required_columns = [
            "T_ambient",
            "T_aanvoer",
            "T_retour",
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

        # --------------------------------------------------------------
        # Ensure numeric measurement columns
        # --------------------------------------------------------------

        numeric_columns = [
            "T_ambient",
            "T_aanvoer",
            "T_retour",
            "T_setpoint",
            "flow_lpm",
            "P_el",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # --------------------------------------------------------------
        # Remove rows without required numeric measurements
        # --------------------------------------------------------------

        df = df.dropna(subset=numeric_columns).copy()

        # --------------------------------------------------------------
        # Water-side temperature difference
        # --------------------------------------------------------------

        df["delta_t_water"] = df["T_aanvoer"] - df["T_retour"]

        # --------------------------------------------------------------
        # Thermal power
        #
        # Flow is L/min:
        #
        # Q_th [kW] =
        #     0.06978 * Flow[L/min] * DeltaT[K]
        # --------------------------------------------------------------

        df["Q_th"] = 0.06978 * df["flow_lpm"] * df["delta_t_water"]

        # --------------------------------------------------------------
        # Measured COP
        # --------------------------------------------------------------

        df["COP_measured"] = df["Q_th"] / df["P_el"]

        # --------------------------------------------------------------
        # Physical sanity checks
        #
        # We only identify the model using points where the heat pump
        # appears to be delivering useful heat.
        # --------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # COP model
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_cop(
        parameters: np.ndarray,
        T_ambient: np.ndarray,
        T_aanvoer: np.ndarray,
    ) -> np.ndarray:
        """
        Predict COP using the semi-physical Carnot model.

        parameters:
            eta_carnot
            delta_t_cond [K]
            delta_t_evap [K]
        """

        eta_carnot, delta_t_cond, delta_t_evap = parameters

        # Approximate refrigerant-side temperatures.
        T_cond_C = T_aanvoer + delta_t_cond

        T_evap_C = T_ambient - delta_t_evap

        # Celsius -> Kelvin
        T_cond_K = T_cond_C + 273.15

        T_evap_K = T_evap_C + 273.15

        temperature_lift = T_cond_K - T_evap_K

        return eta_carnot * T_cond_K / temperature_lift

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        df: pd.DataFrame,
    ) -> HeatPumpCOPModel:
        """
        Calibrate the semi-physical COP model.

        The data is split chronologically:

            first 80%  -> calibration
            last 20%   -> validation

        The validation data is never used during optimization.
        """

        if len(df) < 10:
            raise ValueError("Not enough data points for calibration.")

        # --------------------------------------------------------------
        # Chronological train/test split
        # --------------------------------------------------------------

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 0 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].copy()

        logger.info(
            "Heat pump COP calibration: %d training points, %d validation points",
            len(train_df),
            len(df) - len(train_df),
        )

        # --------------------------------------------------------------
        # Training data
        # --------------------------------------------------------------

        T_ambient = train_df["T_ambient"].to_numpy(dtype=float)

        T_aanvoer = train_df["T_aanvoer"].to_numpy(dtype=float)

        COP_measured = train_df["COP_measured"].to_numpy(dtype=float)

        # --------------------------------------------------------------
        # Optimization objective
        # --------------------------------------------------------------

        def residuals(
            parameters: np.ndarray,
        ) -> np.ndarray:

            COP_predicted = self._predict_cop(
                parameters,
                T_ambient,
                T_aanvoer,
            )

            return COP_predicted - COP_measured

        # --------------------------------------------------------------
        # Initial values
        # --------------------------------------------------------------

        x0 = np.array(
            [
                self.INITIAL_ETA,
                self.INITIAL_DELTA_T_COND,
                self.INITIAL_DELTA_T_EVAP,
            ]
        )

        # --------------------------------------------------------------
        # Parameter bounds
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Fit
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Create model
        # --------------------------------------------------------------

        self.model = HeatPumpCOPModel(
            eta_carnot=eta_carnot,
            delta_t_cond=delta_t_cond,
            delta_t_evap=delta_t_evap,
        )

        return self.model

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Validate the calibrated model on data that was not used
        during calibration.

        Returns:

            r2
            mae
            rmse
        """

        if self.model is None:
            raise RuntimeError("Model must be calibrated before validation.")

        if len(df) < 2:
            raise ValueError("Not enough data points for validation.")

        # --------------------------------------------------------------
        # Same chronological split as calibrate()
        # --------------------------------------------------------------

        split_index = int(len(df) * self.TRAIN_RATIO)

        test_df = df.iloc[split_index:].copy()

        if test_df.empty:
            raise ValueError("No validation data available.")

        # --------------------------------------------------------------
        # Validation inputs
        # --------------------------------------------------------------

        T_ambient = test_df["T_ambient"].to_numpy(dtype=float)

        T_aanvoer = test_df["T_aanvoer"].to_numpy(dtype=float)

        COP_measured = test_df["COP_measured"].to_numpy(dtype=float)

        parameters = np.array(
            [
                self.model.eta_carnot,
                self.model.delta_t_cond,
                self.model.delta_t_evap,
            ]
        )

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        COP_predicted = self._predict_cop(
            parameters,
            T_ambient,
            T_aanvoer,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        logger.info(
            "Heat pump COP validation: R2=%.4f, MAE=%.4f, RMSE=%.4f",
            r2,
            mae,
            rmse,
        )

        # Additional useful logging
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

    # ------------------------------------------------------------------
    # Dataset definition
    # ------------------------------------------------------------------

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
                "T_aanvoer",
                config.heat_pump.supply_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_retour",
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
            # .timeseries(
            #     "T_setpoint",
            #     config.heat_pump.supply_setpoint,
            #     interval="5m",
            #     aggregation="mean",
            #     fill="previous",
            # )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="5m",
                aggregation="last",
                fill="previous",
            )
            .build()
        )
