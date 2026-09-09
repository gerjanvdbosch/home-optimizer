import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from domain.types import (
    BoilerThermalModel,
    Config,
)
from features.dataset import DatasetBuilder, DatasetDefinition
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class BoilerThermalIdentifier(SystemIdentifier[BoilerThermalModel]):
    """
    Three-node stratified boiler thermal model.

    States
    ------
    x[0] = T_top
    x[1] = T_mid
    x[2] = T_bottom

    Measurements
    -----------
    T_top
    T_bottom

    Inputs / disturbances
    ---------------------
    P_heater
    T_cold
    T_ambient
    m_dot

    Additional supervisory signals
    -------------------------------
    T_setpoint
    state

    T_mid is not directly measured. It is estimated as a hidden
    state during model simulation.

    Identification
    --------------
    First 80% of the time series is used for parameter
    identification.

    Last 20% is used exclusively for validation.
    """

    # =========================================================
    # PHYSICAL CONSTANTS
    # =========================================================

    VOLUME_L = 200.0

    RHO = 997.0

    CP = 4180.0

    # ---------------------------------------------------------
    # Three equal volume nodes
    # ---------------------------------------------------------

    VOLUME_M3 = VOLUME_L / 1000.0

    MASS_TOTAL = VOLUME_M3 * RHO

    MASS_NODE = MASS_TOTAL / 3.0

    THERMAL_CAPACITY_NODE = MASS_NODE * CP

    # =========================================================
    # HEATER
    # =========================================================

    # Temporary assumption.
    #
    # Preferably this should eventually be replaced by the
    # actual thermal power calculated from:
    #
    # P = m_dot * cp * (T_aanvoer - T_retour)
    #
    # if those measurements represent the boiler circuit.
    #
    HEATER_POWER_W = 3000.0

    # =========================================================
    # MODEL METADATA
    # =========================================================

    @property
    def name(self) -> str:
        return "boiler"

    @property
    def label(self) -> str:
        return "Boiler Temperature"

    # =========================================================
    # PREPARE
    # =========================================================

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Clean and prepare the boiler identification dataset.
        """

        df = df.copy()

        required_columns = [
            "T_ambient",
            "T_top",
            "T_bottom",
            "T_aanvoer",
            "T_retour",
            "flow_lpm",
            "state",
        ]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing boiler columns: {missing_columns}")

        # -----------------------------------------------------
        # Numeric conversion
        # -----------------------------------------------------

        numeric_columns = [
            "T_ambient",
            "T_top",
            "T_bottom",
            "T_aanvoer",
            "T_retour",
            "flow_lpm",
            "state",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # -----------------------------------------------------
        # Remove impossible temperatures
        # -----------------------------------------------------

        temperature_columns = [
            "T_ambient",
            "T_top",
            "T_bottom",
            "T_aanvoer",
            "T_retour",
        ]

        for column in temperature_columns:
            invalid = ~df[column].between(
                -20.0,
                100.0,
            )

            df.loc[
                invalid,
                column,
            ] = np.nan

        # -----------------------------------------------------
        # Remove impossible flow values
        # -----------------------------------------------------

        df.loc[
            df["flow_lpm"] < 0,
            "flow_lpm",
        ] = np.nan

        # -----------------------------------------------------
        # Sort by time
        # -----------------------------------------------------

        if isinstance(
            df.index,
            pd.DatetimeIndex,
        ):
            df = df.sort_index()

            df = df[~df.index.duplicated(keep="first")]

        # -----------------------------------------------------
        # Forward fill short gaps
        # -----------------------------------------------------

        df = df.ffill()

        # -----------------------------------------------------
        # Only actual boiler temperature measurements are
        # mandatory for identification.
        # -----------------------------------------------------

        df = df.dropna(
            subset=[
                "T_top",
                "T_bottom",
            ]
        )

        if len(df) == 0:
            raise ValueError(
                "No valid boiler temperature data remains after preparation."
            )

        return df

    # =========================================================
    # HEATER POWER
    # =========================================================

    def _heater_power(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert state into a temporary heater power signal.

        Assumption:
            state <= 0 -> heater OFF
            state > 0  -> heater ON

        This should be replaced if the Ecodan state has a
        different meaning.
        """

        state = df["state"].to_numpy(
            dtype=float,
        )

        heater_on = state > 0.0

        return heater_on.astype(float) * self.HEATER_POWER_W

    # =========================================================
    # WATER FLOW
    # =========================================================

    def _mass_flow(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert flow from L/min to kg/s.
        """

        flow_lpm = df["flow_lpm"].to_numpy(
            dtype=float,
        )

        # L/min -> m3/s -> kg/s
        return flow_lpm / 1000.0 / 60.0 * self.RHO

    # =========================================================
    # ODE
    # =========================================================

    def _ode(
        self,
        t: float,
        x: np.ndarray,
        theta: np.ndarray,
        time: np.ndarray,
        P_heater: np.ndarray,
        T_cold: np.ndarray,
        T_ambient: np.ndarray,
        m_dot: np.ndarray,
    ) -> np.ndarray:
        """
        Three-node boiler differential equation.

        x:
            [T_top, T_mid, T_bottom]

        theta:
            [K12, K23, G1, G2, G3, eta]
        """

        # -----------------------------------------------------
        # States
        # -----------------------------------------------------

        T_top = x[0]

        T_mid = x[1]

        T_bottom = x[2]

        # -----------------------------------------------------
        # Parameters
        # -----------------------------------------------------

        (
            K12,
            K23,
            G1,
            G2,
            G3,
            eta,
        ) = theta

        # -----------------------------------------------------
        # Interpolate inputs
        # -----------------------------------------------------

        P = np.interp(
            t,
            time,
            P_heater,
        )

        T_cold_value = np.interp(
            t,
            time,
            T_cold,
        )

        T_ambient_value = np.interp(
            t,
            time,
            T_ambient,
        )

        mass_flow = np.interp(
            t,
            time,
            m_dot,
        )

        # =====================================================
        # HEAT TRANSFER BETWEEN NODES
        # =====================================================

        Q12 = K12 * (T_mid - T_top)

        Q23 = K23 * (T_bottom - T_mid)

        # =====================================================
        # HEAT LOSSES
        # =====================================================

        Q_loss_top = G1 * (T_top - T_ambient_value)

        Q_loss_mid = G2 * (T_mid - T_ambient_value)

        Q_loss_bottom = G3 * (T_bottom - T_ambient_value)

        # =====================================================
        # WATER DRAW
        #
        # Simplified stratified flow:
        #
        # cold water
        #     ↓
        # bottom
        #     ↓
        # middle
        #     ↓
        # top
        #     ↓
        # outlet
        # =====================================================

        Q_flow_top = mass_flow * self.CP * (T_mid - T_top)

        Q_flow_mid = mass_flow * self.CP * (T_bottom - T_mid)

        Q_flow_bottom = mass_flow * self.CP * (T_cold_value - T_bottom)

        # =====================================================
        # ENERGY BALANCES
        # =====================================================

        dT_top = (Q12 - Q_loss_top + Q_flow_top) / self.THERMAL_CAPACITY_NODE

        dT_mid = (Q23 - Q12 - Q_loss_mid + Q_flow_mid) / self.THERMAL_CAPACITY_NODE

        dT_bottom = (
            eta * P - Q23 - Q_loss_bottom + Q_flow_bottom
        ) / self.THERMAL_CAPACITY_NODE

        return np.array(
            [
                dT_top,
                dT_mid,
                dT_bottom,
            ]
        )

    # =========================================================
    # SIMULATE
    # =========================================================

    def _simulate(
        self,
        theta: np.ndarray,
        time: np.ndarray,
        T_top_0: float,
        T_bottom_0: float,
        T_mid_0: float,
        P_heater: np.ndarray,
        T_cold: np.ndarray,
        T_ambient: np.ndarray,
        m_dot: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate the continuous-time boiler model.
        """

        x0 = np.array(
            [
                T_top_0,
                T_mid_0,
                T_bottom_0,
            ]
        )

        solution = solve_ivp(
            lambda t, x: self._ode(
                t=t,
                x=x,
                theta=theta,
                time=time,
                P_heater=P_heater,
                T_cold=T_cold,
                T_ambient=T_ambient,
                m_dot=m_dot,
            ),
            t_span=(
                time[0],
                time[-1],
            ),
            y0=x0,
            t_eval=time,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )

        if not solution.success:
            raise RuntimeError(f"Boiler ODE integration failed: {solution.message}")

        return solution.y

    # =========================================================
    # CALIBRATE
    # =========================================================

    def calibrate(
        self,
        df: pd.DataFrame,
    ) -> BoilerThermalModel:
        """
        Identify boiler parameters.

        80%:
            parameter identification

        20%:
            independent validation
        """

        # =====================================================
        # PREPARE
        # =====================================================

        df = self.prepare(df)

        if len(df) < 20:
            raise ValueError("Not enough samples for boiler identification.")

        if not isinstance(
            df.index,
            pd.DatetimeIndex,
        ):
            raise ValueError("Boiler identification requires a DatetimeIndex.")

        # =====================================================
        # 80 / 20 CHRONOLOGICAL SPLIT
        # =====================================================

        train_ratio = 0.80

        split_index = int(train_ratio * len(df))

        if split_index <= 0:
            raise ValueError("Training dataset is empty.")

        if split_index >= len(df):
            raise ValueError("Validation dataset is empty.")

        df_train = df.iloc[:split_index].copy()

        df_val = df.iloc[split_index:].copy()

        logger.info(
            "Boiler identification dataset: "
            "total=%d, train=%d (%.1f%%), "
            "validation=%d (%.1f%%)",
            len(df),
            len(df_train),
            100.0 * len(df_train) / len(df),
            len(df_val),
            100.0 * len(df_val) / len(df),
        )

        # =====================================================
        # TRAINING DATA
        # =====================================================

        time_train = (df_train.index - df_train.index[0]).total_seconds().to_numpy()

        T_top_train = df_train["T_top"].to_numpy(
            dtype=float,
        )

        T_bottom_train = df_train["T_bottom"].to_numpy(
            dtype=float,
        )

        P_train = self._heater_power(df_train)

        T_ambient_train = df_train["T_ambient"].to_numpy(
            dtype=float,
        )

        # -----------------------------------------------------
        # TEMPORARY ASSUMPTION:
        #
        # return temperature used as cold-side temperature.
        #
        # This should be replaced with actual cold-water
        # inlet temperature when available.
        # -----------------------------------------------------

        T_cold_train = df_train["T_retour"].to_numpy(
            dtype=float,
        )

        m_dot_train = self._mass_flow(df_train)

        # =====================================================
        # INITIAL MID TEMPERATURE
        # =====================================================

        T_mid_initial = (T_top_train[0] + T_bottom_train[0]) / 2.0

        # =====================================================
        # INITIAL PARAMETER GUESS
        # =====================================================

        # theta =
        #
        # [K12, K23, G1, G2, G3, eta, Tmid0]
        #
        # K12/K23 = W/K
        # G1/G2/G3 = W/K
        # eta = -
        # Tmid0 = degC

        theta_0 = np.array(
            [
                5.0,
                5.0,
                1.0,
                1.0,
                1.0,
                0.95,
                T_mid_initial,
            ]
        )

        # =====================================================
        # PARAMETER BOUNDS
        # =====================================================

        lower_bounds = np.array(
            [
                0.001,  # K12
                0.001,  # K23
                0.0,  # G1
                0.0,  # G2
                0.0,  # G3
                0.5,  # eta
                0.0,  # Tmid0
            ]
        )

        upper_bounds = np.array(
            [
                100.0,  # K12
                100.0,  # K23
                20.0,  # G1
                20.0,  # G2
                20.0,  # G3
                1.0,  # eta
                100.0,  # Tmid0
            ]
        )

        # =====================================================
        # RESIDUAL FUNCTION
        # =====================================================

        def residual(
            theta_full: np.ndarray,
        ) -> np.ndarray:
            """
            Residuals used by least_squares.

            Only measured states are compared:

                T_top
                T_bottom

            T_mid remains a hidden state.
            """

            theta = theta_full[:6]

            T_mid_0 = theta_full[6]

            try:
                x_pred = self._simulate(
                    theta=theta,
                    time=time_train,
                    T_top_0=T_top_train[0],
                    T_bottom_0=(T_bottom_train[0]),
                    T_mid_0=T_mid_0,
                    P_heater=P_train,
                    T_cold=T_cold_train,
                    T_ambient=(T_ambient_train),
                    m_dot=m_dot_train,
                )

            except RuntimeError:
                return np.ones(2 * len(time_train)) * 1e6

            # -------------------------------------------------
            # Measurement residuals
            # -------------------------------------------------

            error_top = x_pred[0] - T_top_train

            error_bottom = x_pred[2] - T_bottom_train

            return np.concatenate(
                [
                    error_top,
                    error_bottom,
                ]
            )

        # =====================================================
        # IDENTIFICATION
        # =====================================================

        logger.info("Starting boiler parameter identification...")

        result = least_squares(
            residual,
            x0=theta_0,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            method="trf",
            max_nfev=200,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )

        if not result.success:
            logger.warning(
                "Boiler parameter identification did not fully converge: %s",
                result.message,
            )

        # =====================================================
        # IDENTIFIED PARAMETERS
        # =====================================================

        identified = result.x

        K12 = identified[0]

        K23 = identified[1]

        G1 = identified[2]

        G2 = identified[3]

        G3 = identified[4]

        eta = identified[5]

        T_mid_0 = identified[6]

        logger.info("Identified boiler parameters:")

        logger.info(
            "K12=%.6f W/K",
            K12,
        )

        logger.info(
            "K23=%.6f W/K",
            K23,
        )

        logger.info(
            "G1=%.6f W/K",
            G1,
        )

        logger.info(
            "G2=%.6f W/K",
            G2,
        )

        logger.info(
            "G3=%.6f W/K",
            G3,
        )

        logger.info(
            "eta=%.6f",
            eta,
        )

        logger.info(
            "T_mid_0=%.3f °C",
            T_mid_0,
        )

        logger.info(
            "Optimization cost=%.6f",
            result.cost,
        )

        # =====================================================
        # TRAINING SIMULATION
        # =====================================================

        theta_model = identified[:6]

        x_train_pred = self._simulate(
            theta=theta_model,
            time=time_train,
            T_top_0=T_top_train[0],
            T_bottom_0=(T_bottom_train[0]),
            T_mid_0=T_mid_0,
            P_heater=P_train,
            T_cold=T_cold_train,
            T_ambient=T_ambient_train,
            m_dot=m_dot_train,
        )

        # =====================================================
        # TRAINING RMSE
        # =====================================================

        train_top_error = x_train_pred[0] - T_top_train

        train_bottom_error = x_train_pred[2] - T_bottom_train

        train_rmse_top = np.sqrt(np.mean(train_top_error**2))

        train_rmse_bottom = np.sqrt(np.mean(train_bottom_error**2))

        logger.info(
            "Training RMSE: top=%.3f °C, bottom=%.3f °C",
            train_rmse_top,
            train_rmse_bottom,
        )

        # =====================================================
        # VALIDATION DATA
        # =====================================================

        time_val = (df_val.index - df_val.index[0]).total_seconds().to_numpy()

        T_top_val = df_val["T_top"].to_numpy(
            dtype=float,
        )

        T_bottom_val = df_val["T_bottom"].to_numpy(
            dtype=float,
        )

        P_val = self._heater_power(df_val)

        T_ambient_val = df_val["T_ambient"].to_numpy(
            dtype=float,
        )

        T_cold_val = df_val["T_retour"].to_numpy(
            dtype=float,
        )

        m_dot_val = self._mass_flow(df_val)

        # =====================================================
        # VALIDATION INITIAL STATE
        # =====================================================

        # T_top and T_bottom are known at the beginning of
        # validation.
        #
        # T_mid is not measured.
        #
        # For an independent validation run we therefore use
        # a neutral estimate.
        #
        # Later this can be replaced by an observer.
        #

        T_mid_val_0 = (T_top_val[0] + T_bottom_val[0]) / 2.0

        # =====================================================
        # VALIDATION SIMULATION
        # =====================================================

        x_val_pred = self._simulate(
            theta=theta_model,
            time=time_val,
            T_top_0=T_top_val[0],
            T_bottom_0=(T_bottom_val[0]),
            T_mid_0=T_mid_val_0,
            P_heater=P_val,
            T_cold=T_cold_val,
            T_ambient=T_ambient_val,
            m_dot=m_dot_val,
        )

        # =====================================================
        # VALIDATION RMSE
        # =====================================================

        val_top_error = x_val_pred[0] - T_top_val

        val_bottom_error = x_val_pred[2] - T_bottom_val

        validation_rmse_top = np.sqrt(np.mean(val_top_error**2))

        validation_rmse_bottom = np.sqrt(np.mean(val_bottom_error**2))

        logger.info(
            "Validation RMSE: top=%.3f °C, bottom=%.3f °C",
            validation_rmse_top,
            validation_rmse_bottom,
        )

        # =====================================================
        # STORE MODEL
        # =====================================================

        self.model = BoilerThermalModel(
            K12=K12,
            K23=K23,
            G1=G1,
            G2=G2,
            G3=G3,
            eta=eta,
            T_mid_0=T_mid_0,
            rmse_top=train_rmse_top,
            rmse_bottom=train_rmse_bottom,
            validation_rmse_top=(validation_rmse_top),
            validation_rmse_bottom=(validation_rmse_bottom),
        )

        return self.model

    # =========================================================
    # DATASET
    # =========================================================

    def dataset(
        self,
        config: Config,
    ) -> DatasetDefinition:

        return (
            DatasetBuilder()
            # -------------------------------------------------
            # Ambient
            # -------------------------------------------------
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Boiler top
            # -------------------------------------------------
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Boiler bottom
            # -------------------------------------------------
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Heat pump supply
            # -------------------------------------------------
            .timeseries(
                "T_aanvoer",
                config.heat_pump.supply_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Heat pump return
            # -------------------------------------------------
            .timeseries(
                "T_retour",
                config.heat_pump.return_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Flow
            # -------------------------------------------------
            .timeseries(
                "flow_lpm",
                config.heat_pump.flow,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            # -------------------------------------------------
            # Heat pump state
            # -------------------------------------------------
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="5m",
                aggregation="last",
                fill="previous",
            )
            # -------------------------------------------------
            # Boiler setpoint
            #
            # This is included as supervisory/control
            # information.
            #
            # It is NOT directly used in the thermal ODE.
            # -------------------------------------------------
            # .timeseries(
            #     "T_setpoint",
            #     config.heat_pump.boiler.setpoint,
            #     interval="5m",
            #     aggregation="last",
            #     fill="previous",
            # )
            .build()
        )
