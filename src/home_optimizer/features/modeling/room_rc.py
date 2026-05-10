from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

LOGGER = logging.getLogger(__name__)

MODEL_TYPE = "physical_2state_rc"
STATE_DIM = 2
INPUT_COLUMNS = (
    "outdoor_temp_c",
    "heating_kw_eff",
    "solar_glass_kw",
    "solar_glass_kw_filtered",
    "occupied_flag",
    "hour_sin",
    "hour_cos",
)
FORECAST_FEATURE_COLUMNS = (
    "timestamp",
    "outdoor_temp_c",
    "heating_kw",
    "irradiance_wm2",
    "shutter_position",
    "occupied_flag",
)
TRAIN_REQUIRED_COLUMNS = ("room_temp_c",) + FORECAST_FEATURE_COLUMNS
DEFAULT_SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "sunny": "solar irradiance >= 150 W/m2",
    "heating_active": "heating input >= 0.40 kW",
    "cold_weather": "outdoor temperature <= 5.0 C",
    "freezing_weather": "outdoor temperature <= 0.0 C",
    "mild_weather": "outdoor temperature >= 12.0 C",
    "freezing_and_heating": "outdoor temperature <= 0.0 C and heating >= 0.40 kW",
    "freezing_night": "outdoor temperature <= 0.0 C during night hours",
    "shutters_open": "shutter factor >= 0.75",
    "shutters_closed": "shutter factor <= 0.25",
    "sunny_shutters_open": "irradiance >= 150 W/m2 and shutter factor >= 0.75",
    "sunny_shutters_closed": "irradiance >= 150 W/m2 and shutter factor <= 0.25",
    "heating_and_sunny": "heating >= 0.40 kW and irradiance >= 150 W/m2",
    "night": "local hour >= 22 or local hour < 6",
    "occupied": "occupied flag == 1",
    "sunny_midday": "irradiance >= 150 W/m2 and 11:00 <= local hour < 16:00",
}


@dataclass
class RoomRC2StateConfig:
    interval_minutes: int = 10
    glass_area_m2: float = 8.0
    g_glass: float = 0.50
    shutter_mode: str = "open_percent"
    alpha_solar: float = 0.85
    alpha_heat: float = 0.70
    use_segment_weights: bool = True
    use_huber_loss: bool = True
    huber_delta_c: float = 0.25
    kalman_process_noise_air: float = 0.02
    kalman_process_noise_mass: float = 0.005
    kalman_measurement_noise: float = 0.03
    max_forecast_temp_c: float = 45.0
    min_forecast_temp_c: float = -10.0
    local_timezone: Optional[str] = None
    gap_factor: float = 1.5
    short_gap_interpolation_limit: int = 2
    min_train_rows: int = 96
    optimizer_maxiter: int = 300
    spectral_radius_penalty: float = 1e5
    physical_penalty_weight: float = 1e3
    regularization_weight: float = 1e-3
    train_weight_freezing: float = 2.0
    train_weight_heating_active: float = 1.0
    train_weight_sunny_shutters_open: float = 1.0
    train_weight_night: float = 0.5
    sunny_irradiance_threshold_wm2: float = 150.0
    heating_active_threshold_kw: float = 0.40
    shutters_open_threshold: float = 0.75
    shutters_closed_threshold: float = 0.25
    cold_outdoor_temp_threshold_c: float = 5.0
    freezing_outdoor_temp_threshold_c: float = 0.0
    mild_outdoor_temp_threshold_c: float = 12.0
    night_start_hour: int = 22
    night_end_hour: int = 6
    sunny_midday_start_hour: int = 11
    sunny_midday_end_hour: int = 16
    R_air_out_min: float = 0.1
    R_air_out_max: float = 100.0
    R_air_mass_min: float = 0.1
    R_air_mass_max: float = 100.0
    R_mass_out_min: float = 0.1
    R_mass_out_max: float = 100.0
    C_air_min: float = 0.05
    C_air_max: float = 10.0
    C_mass_min: float = 1.0
    C_mass_max: float = 500.0
    eta_heat_min: float = 0.1
    eta_heat_max: float = 1.5
    eta_solar_air_min: float = 0.0
    eta_solar_air_max: float = 1.5
    eta_solar_mass_min: float = 0.0
    eta_solar_mass_max: float = 1.5
    eta_internal_min: float = -0.2
    eta_internal_max: float = 0.5
    hour_coeff_min: float = -0.5
    hour_coeff_max: float = 0.5
    initial_mass_offset_min: float = -5.0
    initial_mass_offset_max: float = 5.0

    def __post_init__(self) -> None:
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        if self.shutter_mode not in {"open_percent", "closed_percent"}:
            raise ValueError("shutter_mode must be 'open_percent' or 'closed_percent'")
        self.alpha_solar = float(np.clip(self.alpha_solar, 0.0, 0.99))
        self.alpha_heat = float(np.clip(self.alpha_heat, 0.0, 0.99))
        if self.g_glass < 0.0:
            raise ValueError("g_glass must be non-negative")
        if self.glass_area_m2 < 0.0:
            raise ValueError("glass_area_m2 must be non-negative")
        if self.huber_delta_c <= 0.0:
            raise ValueError("huber_delta_c must be positive")

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (self.R_air_out_min, self.R_air_out_max),
            (self.R_air_mass_min, self.R_air_mass_max),
            (self.R_mass_out_min, self.R_mass_out_max),
            (self.C_air_min, self.C_air_max),
            (self.C_mass_min, self.C_mass_max),
            (self.eta_heat_min, self.eta_heat_max),
            (self.eta_solar_air_min, self.eta_solar_air_max),
            (self.eta_solar_mass_min, self.eta_solar_mass_max),
            (self.eta_internal_min, self.eta_internal_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            (self.initial_mass_offset_min, self.initial_mass_offset_max),
        ]


@dataclass
class RoomRC2StateParams:
    R_air_out: float = 5.0
    R_air_mass: float = 1.0
    R_mass_out: float = 20.0
    C_air: float = 0.5
    C_mass: float = 50.0
    eta_heat: float = 0.9
    eta_solar_air: float = 0.2
    eta_solar_mass: float = 0.5
    eta_internal: float = 0.1
    b_hour_sin_air: float = 0.0
    b_hour_cos_air: float = 0.0
    b_hour_sin_mass: float = 0.0
    b_hour_cos_mass: float = 0.0
    initial_mass_offset_c: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.R_air_out,
                self.R_air_mass,
                self.R_mass_out,
                self.C_air,
                self.C_mass,
                self.eta_heat,
                self.eta_solar_air,
                self.eta_solar_mass,
                self.eta_internal,
                self.b_hour_sin_air,
                self.b_hour_cos_air,
                self.b_hour_sin_mass,
                self.b_hour_cos_mass,
                self.initial_mass_offset_c,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "RoomRC2StateParams":
        values = [float(value) for value in vector]
        return cls(*values)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoomRC2StateParams":
        return cls(**{key: float(value) for key, value in data.items()})


@dataclass
class _SequenceCache:
    frame: pd.DataFrame
    inputs: np.ndarray
    measurements: np.ndarray
    sample_weights: np.ndarray
    segment_masks: dict[str, np.ndarray]


class RoomRC2StatePhysicalModel:
    """Physical 2-state RC grey-box model for room temperature forecasting."""

    def __init__(self, config: RoomRC2StateConfig):
        self.config = config
        self.params = RoomRC2StateParams()
        self.training_metadata: dict[str, Any] = {}
        self.created_at_utc = datetime.now(timezone.utc)
        self.feature_schema = {
            "train_required_columns": list(TRAIN_REQUIRED_COLUMNS),
            "forecast_required_columns": list(FORECAST_FEATURE_COLUMNS),
            "input_columns": list(INPUT_COLUMNS),
        }

    def get_params(self) -> RoomRC2StateParams:
        return RoomRC2StateParams.from_dict(self.params.to_dict())

    def set_params(self, params: RoomRC2StateParams) -> None:
        self.params = RoomRC2StateParams.from_dict(params.to_dict())

    def save(self, path: str) -> None:
        payload = {
            "model_type": MODEL_TYPE,
            "created_at_utc": self.created_at_utc.isoformat(),
            "interval_minutes": self.config.interval_minutes,
            "config": asdict(self.config),
            "params": self.params.to_dict(),
            "training_metadata": self.training_metadata,
            "feature_schema": self.feature_schema,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "RoomRC2StatePhysicalModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {payload.get('model_type')}")
        model = cls(RoomRC2StateConfig(**payload["config"]))
        model.params = RoomRC2StateParams.from_dict(payload["params"])
        model.training_metadata = payload.get("training_metadata", {})
        model.feature_schema = payload.get("feature_schema", model.feature_schema)
        created = payload.get("created_at_utc")
        if created is not None:
            model.created_at_utc = datetime.fromisoformat(created)
        return model

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_features(df, require_room_temp="room_temp_c" in df.columns)

    def split_into_sequences(self, df_prepared: pd.DataFrame) -> list[pd.DataFrame]:
        if df_prepared.empty:
            return []
        diffs_minutes = (
            df_prepared["timestamp"].diff().dt.total_seconds().fillna(0.0) / 60.0
        )
        gap_threshold = self.config.gap_factor * self.config.interval_minutes
        sequence_id = (diffs_minutes > gap_threshold).cumsum()
        sequences = [part.reset_index(drop=True) for _, part in df_prepared.groupby(sequence_id)]
        LOGGER.info("Split prepared data into %s sequences", len(sequences))
        return sequences

    def params_to_matrices(
        self, params: RoomRC2StateParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c_air = params.C_air
        c_mass = params.C_mass
        r_air_out = params.R_air_out
        r_air_mass = params.R_air_mass
        r_mass_out = params.R_mass_out

        F = np.array(
            [
                [
                    -((1.0 / r_air_out) + (1.0 / r_air_mass)) / c_air,
                    (1.0 / r_air_mass) / c_air,
                ],
                [
                    (1.0 / r_air_mass) / c_mass,
                    -((1.0 / r_air_mass) + (1.0 / r_mass_out)) / c_mass,
                ],
            ],
            dtype=float,
        )
        G = np.array(
            [
                [
                    (1.0 / r_air_out) / c_air,
                    params.eta_heat / c_air,
                    params.eta_solar_air / c_air,
                    0.0,
                    params.eta_internal / c_air,
                    params.b_hour_sin_air / c_air,
                    params.b_hour_cos_air / c_air,
                ],
                [
                    (1.0 / r_mass_out) / c_mass,
                    0.0,
                    0.0,
                    params.eta_solar_mass / c_mass,
                    0.0,
                    params.b_hour_sin_mass / c_mass,
                    params.b_hour_cos_mass / c_mass,
                ],
            ],
            dtype=float,
        )
        top = np.hstack([F, G])
        bottom = np.zeros((len(INPUT_COLUMNS), STATE_DIM + len(INPUT_COLUMNS)), dtype=float)
        M = np.vstack([top, bottom])
        expM = expm(M * self.config.dt_hours)
        A_d = expM[:STATE_DIM, :STATE_DIM]
        B_d = expM[:STATE_DIM, STATE_DIM:]
        return F, G, A_d, B_d

    def fit(
        self,
        df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (1, 6, 36, 72),
        include_144: bool = False,
    ) -> dict[str, Any]:
        prepared = self._prepare_features(df, require_room_temp=True)
        sequences = self.split_into_sequences(prepared)
        sample_count = int(sum(len(sequence) for sequence in sequences))
        if sample_count < self.config.min_train_rows:
            raise ValueError(
                f"Training data has {sample_count} rows, fewer than min_train_rows={self.config.min_train_rows}"
            )

        horizon_list = self._normalize_horizons(horizons, include_144=include_144)
        weights = self._horizon_weights(horizon_list)
        caches = [self._build_sequence_cache(sequence) for sequence in sequences]
        self._log_training_context(prepared, caches)

        def objective(vector: np.ndarray) -> float:
            params = RoomRC2StateParams.from_vector(vector)
            loss = self._objective_for_sequences(caches, params, horizon_list, weights)
            if not np.isfinite(loss):
                return 1e12
            return float(loss)

        x0 = self.params.to_vector()
        result = minimize(
            objective,
            x0=x0,
            bounds=self.config.bounds(),
            method="L-BFGS-B",
            options={"maxiter": self.config.optimizer_maxiter},
        )
        fitted_params = RoomRC2StateParams.from_vector(result.x)
        self.params = fitted_params
        train_loss = float(objective(result.x))

        validation_metrics = None
        if validation_df is not None:
            validation_metrics = self.evaluate(validation_df, horizons=self._normalize_horizons(horizon_list, include_144=False))

        self.training_metadata = {
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_sample_count": sample_count,
            "train_sequence_count": len(sequences),
            "horizons": list(horizon_list),
            "horizon_weights": {str(k): v for k, v in weights.items()},
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(getattr(result, "nit", 0)),
            "final_train_loss": train_loss,
            "validation_metrics": validation_metrics,
        }
        LOGGER.info("Fitted RC parameters: %s", self.params.to_dict())
        LOGGER.info("Final train loss: %.6f", train_loss)
        if validation_metrics is not None:
            LOGGER.info("Validation metrics: %s", validation_metrics)
        return {
            "model_type": MODEL_TYPE,
            "train_sample_count": sample_count,
            "train_sequence_count": len(sequences),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "final_train_loss": train_loss,
            "params": self.params.to_dict(),
            "validation_metrics": validation_metrics,
        }

    def predict_one_step(
        self, df: pd.DataFrame, initial_state: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        prepared = self._prepare_features(df, require_room_temp=True)
        sequences = self.split_into_sequences(prepared)
        rows: list[dict[str, Any]] = []
        for sequence in sequences:
            cache = self._build_sequence_cache(sequence)
            state = (
                np.asarray(initial_state, dtype=float).copy()
                if initial_state is not None
                else self._initial_state_from_sequence(sequence, self.params)
            )
            covariance = self._initial_covariance()
            rows.append(
                {
                    "timestamp": sequence.iloc[0]["timestamp"],
                    "room_temp_c": float(sequence.iloc[0]["room_temp_c"]),
                    "predicted_room_temp_c": float(state[0]),
                    "predicted_mass_temp_c": float(state[1]),
                    "outdoor_temp_c": float(sequence.iloc[0]["outdoor_temp_c"]),
                    "heating_kw": float(sequence.iloc[0]["heating_kw"]),
                    "heating_kw_eff": float(sequence.iloc[0]["heating_kw_eff"]),
                    "irradiance_wm2": float(sequence.iloc[0]["irradiance_wm2"]),
                    "shutter_factor": float(sequence.iloc[0]["shutter_factor"]),
                    "solar_glass_kw": float(sequence.iloc[0]["solar_glass_kw"]),
                    "solar_glass_kw_filtered": float(sequence.iloc[0]["solar_glass_kw_filtered"]),
                }
            )
            for index in range(1, len(sequence)):
                row = sequence.iloc[index]
                u_prev = cache.inputs[index - 1]
                measurement = float(row["room_temp_c"])
                state, covariance = self.filter_update(state, covariance, u_prev, measurement)
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "room_temp_c": measurement,
                        "predicted_room_temp_c": float(state[0]),
                        "predicted_mass_temp_c": float(state[1]),
                        "outdoor_temp_c": float(row["outdoor_temp_c"]),
                        "heating_kw": float(row["heating_kw"]),
                        "heating_kw_eff": float(row["heating_kw_eff"]),
                        "irradiance_wm2": float(row["irradiance_wm2"]),
                        "shutter_factor": float(row["shutter_factor"]),
                        "solar_glass_kw": float(row["solar_glass_kw"]),
                        "solar_glass_kw_filtered": float(row["solar_glass_kw_filtered"]),
                    }
                )
        return pd.DataFrame(rows)

    def forecast(
        self,
        future_df: pd.DataFrame,
        current_state: Optional[np.ndarray] = None,
        current_covariance: Optional[np.ndarray] = None,
        last_measurement_c: Optional[float] = None,
        horizon_steps: int = 72,
    ) -> pd.DataFrame:
        prepared = self._prepare_features(future_df, require_room_temp=False)
        if prepared.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "horizon_steps",
                    "horizon_minutes",
                    "predicted_room_temp_c",
                    "predicted_mass_temp_c",
                    "outdoor_temp_c",
                    "heating_kw",
                    "heating_kw_eff",
                    "irradiance_wm2",
                    "shutter_factor",
                    "solar_glass_kw",
                    "solar_glass_kw_filtered",
                ]
            )
        if current_state is None:
            if "room_temp_c" not in prepared.columns:
                raise ValueError("current_state is required when future_df has no room_temp_c column")
            current_state = self._initial_state_from_sequence(prepared, self.params)
        state = np.asarray(current_state, dtype=float).copy()
        covariance = (
            np.asarray(current_covariance, dtype=float).copy()
            if current_covariance is not None
            else self._initial_covariance()
        )
        inputs = prepared.loc[:, INPUT_COLUMNS].to_numpy(dtype=float)
        if last_measurement_c is not None:
            state, covariance = self.filter_update(state, covariance, inputs[0], float(last_measurement_c))
        _, _, A_d, B_d = self.params_to_matrices(self.params)

        rows: list[dict[str, Any]] = []
        limit = min(horizon_steps, len(prepared))
        for step in range(limit):
            row = prepared.iloc[step]
            u = inputs[step]
            state = A_d @ state + B_d @ u
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "horizon_steps": step + 1,
                    "horizon_minutes": int((step + 1) * self.config.interval_minutes),
                    "predicted_room_temp_c": float(state[0]),
                    "predicted_mass_temp_c": float(state[1]),
                    "outdoor_temp_c": float(row["outdoor_temp_c"]),
                    "heating_kw": float(row["heating_kw"]),
                    "heating_kw_eff": float(row["heating_kw_eff"]),
                    "irradiance_wm2": float(row["irradiance_wm2"]),
                    "shutter_factor": float(row["shutter_factor"]),
                    "solar_glass_kw": float(row["solar_glass_kw"]),
                    "solar_glass_kw_filtered": float(row["solar_glass_kw_filtered"]),
                }
            )
        return pd.DataFrame(rows)

    def filter_update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        u: np.ndarray,
        measurement_room_temp_c: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, A_d, B_d = self.params_to_matrices(self.params)
        x_pred = A_d @ state + B_d @ u
        Q = np.diag(
            [
                self.config.kalman_process_noise_air**2,
                self.config.kalman_process_noise_mass**2,
            ]
        )
        P_pred = A_d @ covariance @ A_d.T + Q
        if measurement_room_temp_c is None or not np.isfinite(measurement_room_temp_c):
            return x_pred, P_pred

        C = np.array([[1.0, 0.0]], dtype=float)
        R = np.array([[self.config.kalman_measurement_noise**2]], dtype=float)
        innovation = np.array([[measurement_room_temp_c]], dtype=float) - (C @ x_pred.reshape(-1, 1))
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_upd = x_pred.reshape(-1, 1) + K @ innovation
        P_upd = (np.eye(STATE_DIM) - K @ C) @ P_pred
        return x_upd.ravel(), P_upd

    def evaluate(
        self,
        df: pd.DataFrame,
        horizons: Sequence[int] = (1, 6, 36, 72, 144),
    ) -> dict[str, Any]:
        prepared = self._prepare_features(df, require_room_temp=True)
        sequences = self.split_into_sequences(prepared)
        caches = [self._build_sequence_cache(sequence) for sequence in sequences]
        horizon_list = self._normalize_horizons(horizons, include_144=False)
        errors_by_horizon = self._collect_errors(caches, self.params, horizon_list)
        metrics = [self._metric_dict(h, errors_by_horizon[h]) for h in horizon_list]
        return {"aggregate_metrics": metrics}

    def evaluate_segments(
        self,
        df: pd.DataFrame,
        horizons: Sequence[int] = (1, 6, 36, 72, 144),
    ) -> dict[str, Any]:
        prepared = self._prepare_features(df, require_room_temp=True)
        sequences = self.split_into_sequences(prepared)
        caches = [self._build_sequence_cache(sequence) for sequence in sequences]
        horizon_list = self._normalize_horizons(horizons, include_144=False)
        segment_errors = self._collect_segment_errors(caches, self.params, horizon_list)

        reports = []
        for name, description in DEFAULT_SEGMENT_DESCRIPTIONS.items():
            reports.append(
                {
                    "segment_name": name,
                    "description": description,
                    "metrics": [self._metric_dict(h, segment_errors[name][h]) for h in horizon_list],
                }
            )
        return {"segment_metrics": reports}

    def _prepare_features(self, df: pd.DataFrame, require_room_temp: bool) -> pd.DataFrame:
        required_columns = set(FORECAST_FEATURE_COLUMNS)
        if require_room_temp:
            required_columns.add("room_temp_c")
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = df.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if frame["timestamp"].isna().any():
            raise ValueError("timestamp contains unparseable values")
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        essential_numeric = [
            column
            for column in ["room_temp_c", "outdoor_temp_c", "heating_kw", "irradiance_wm2", "shutter_position", "occupied_flag"]
            if column in frame.columns
        ]
        missing_before = frame[essential_numeric].isna().sum().to_dict()
        if any(count > 0 for count in missing_before.values()):
            LOGGER.warning("Missing values before interpolation: %s", missing_before)

        if essential_numeric:
            frame.loc[:, essential_numeric] = frame.loc[:, essential_numeric].interpolate(
                method="linear",
                limit=self.config.short_gap_interpolation_limit,
                limit_direction="both",
            )

        missing_after = frame[essential_numeric].isna().sum()
        if int(missing_after.sum()) > 0:
            raise ValueError(f"Missing values remain after interpolation: {missing_after.to_dict()}")

        frame["heating_kw"] = frame["heating_kw"].clip(lower=0.0)
        frame["irradiance_wm2"] = frame["irradiance_wm2"].clip(lower=0.0)
        frame["occupied_flag"] = frame["occupied_flag"].clip(lower=0.0, upper=1.0)
        frame["shutter_factor"] = frame["shutter_position"].map(self._shutter_factor_from_position)
        frame["solar_glass_kw"] = (
            frame["irradiance_wm2"]
            * self.config.glass_area_m2
            * self.config.g_glass
            * frame["shutter_factor"]
            / 1000.0
        )
        frame["solar_glass_kw_filtered"] = self._exp_filter(
            frame["solar_glass_kw"].to_numpy(dtype=float),
            alpha=self.config.alpha_solar,
        )
        frame["heating_kw_eff"] = self._exp_filter(
            frame["heating_kw"].to_numpy(dtype=float),
            alpha=self.config.alpha_heat,
        )

        ts_local = frame["timestamp"]
        if self.config.local_timezone is not None:
            ts_local = ts_local.dt.tz_convert(self.config.local_timezone)
        local_hour = ts_local.dt.hour + (ts_local.dt.minute / 60.0)
        frame["local_hour"] = local_hour.astype(float)
        angle = 2.0 * math.pi * frame["local_hour"] / 24.0
        frame["hour_sin"] = np.sin(angle)
        frame["hour_cos"] = np.cos(angle)

        diffs = frame["timestamp"].diff().dt.total_seconds().dropna() / 60.0
        if not diffs.empty:
            irregular = diffs[np.abs(diffs - self.config.interval_minutes) > 1e-6]
            if not irregular.empty:
                LOGGER.warning(
                    "Detected %s irregular intervals; expected %s minutes",
                    len(irregular),
                    self.config.interval_minutes,
                )

        LOGGER.info(
            "Prepared %s samples spanning %s to %s",
            len(frame),
            frame["timestamp"].iloc[0],
            frame["timestamp"].iloc[-1],
        )
        return frame

    def _shutter_factor_from_position(self, shutter_position: float) -> float:
        raw = float(shutter_position)
        if self.config.shutter_mode == "open_percent":
            factor = raw / 100.0
        else:
            factor = 1.0 - (raw / 100.0)
        return float(np.clip(factor, 0.0, 1.0))

    def _exp_filter(self, values: np.ndarray, alpha: float) -> np.ndarray:
        if len(values) == 0:
            return np.array([], dtype=float)
        filtered = np.zeros_like(values, dtype=float)
        filtered[0] = float(values[0])
        for index in range(1, len(values)):
            filtered[index] = (alpha * filtered[index - 1]) + ((1.0 - alpha) * values[index])
        return filtered

    def _normalize_horizons(
        self, horizons: Sequence[int], include_144: bool
    ) -> tuple[int, ...]:
        normalized = sorted({int(h) for h in horizons if int(h) > 0})
        if include_144 and 144 not in normalized:
            normalized.append(144)
        return tuple(sorted(normalized))

    def _horizon_weights(self, horizons: Sequence[int]) -> dict[int, float]:
        if tuple(horizons) == (1, 6, 36, 72):
            return {1: 0.20, 6: 0.30, 36: 0.30, 72: 0.20}
        if tuple(horizons) == (1, 6, 36, 72, 144):
            return {1: 0.15, 6: 0.25, 36: 0.25, 72: 0.25, 144: 0.10}
        uniform = 1.0 / max(1, len(horizons))
        return {int(h): uniform for h in horizons}

    def _build_sequence_cache(self, sequence: pd.DataFrame) -> _SequenceCache:
        segment_masks = self._segment_masks(sequence)
        weights = self._training_weights(segment_masks, len(sequence))
        return _SequenceCache(
            frame=sequence,
            inputs=sequence.loc[:, INPUT_COLUMNS].to_numpy(dtype=float),
            measurements=sequence["room_temp_c"].to_numpy(dtype=float),
            sample_weights=weights,
            segment_masks=segment_masks,
        )

    def _training_weights(self, segment_masks: dict[str, np.ndarray], length: int) -> np.ndarray:
        weights = np.ones(length, dtype=float)
        if not self.config.use_segment_weights:
            return weights
        weights = weights + (self.config.train_weight_freezing * segment_masks["freezing_weather"])
        weights = weights + (self.config.train_weight_heating_active * segment_masks["heating_active"])
        weights = weights + (
            self.config.train_weight_sunny_shutters_open * segment_masks["sunny_shutters_open"]
        )
        weights = weights + (self.config.train_weight_night * segment_masks["night"])
        return weights.astype(float)

    def _segment_masks(self, sequence: pd.DataFrame) -> dict[str, np.ndarray]:
        local_hour = sequence["local_hour"].to_numpy(dtype=float)
        night = (local_hour >= self.config.night_start_hour) | (local_hour < self.config.night_end_hour)
        sunny = sequence["irradiance_wm2"].to_numpy(dtype=float) >= self.config.sunny_irradiance_threshold_wm2
        heating_active = sequence["heating_kw"].to_numpy(dtype=float) >= self.config.heating_active_threshold_kw
        freezing_weather = sequence["outdoor_temp_c"].to_numpy(dtype=float) <= self.config.freezing_outdoor_temp_threshold_c
        cold_weather = sequence["outdoor_temp_c"].to_numpy(dtype=float) <= self.config.cold_outdoor_temp_threshold_c
        mild_weather = sequence["outdoor_temp_c"].to_numpy(dtype=float) >= self.config.mild_outdoor_temp_threshold_c
        shutters_open = sequence["shutter_factor"].to_numpy(dtype=float) >= self.config.shutters_open_threshold
        shutters_closed = sequence["shutter_factor"].to_numpy(dtype=float) <= self.config.shutters_closed_threshold
        occupied = sequence["occupied_flag"].to_numpy(dtype=float) >= 0.5
        sunny_midday = sunny & (
            (local_hour >= self.config.sunny_midday_start_hour)
            & (local_hour < self.config.sunny_midday_end_hour)
        )
        return {
            "sunny": sunny,
            "heating_active": heating_active,
            "cold_weather": cold_weather,
            "freezing_weather": freezing_weather,
            "mild_weather": mild_weather,
            "freezing_and_heating": freezing_weather & heating_active,
            "freezing_night": freezing_weather & night,
            "shutters_open": shutters_open,
            "shutters_closed": shutters_closed,
            "sunny_shutters_open": sunny & shutters_open,
            "sunny_shutters_closed": sunny & shutters_closed,
            "heating_and_sunny": heating_active & sunny,
            "night": night,
            "occupied": occupied,
            "sunny_midday": sunny_midday,
        }

    def _initial_state_from_sequence(
        self, sequence: pd.DataFrame, params: RoomRC2StateParams
    ) -> np.ndarray:
        air = float(sequence["room_temp_c"].iloc[0])
        count = min(36, len(sequence))
        mass = float(sequence["room_temp_c"].iloc[:count].mean()) + params.initial_mass_offset_c
        mass = float(np.clip(mass, self.config.min_forecast_temp_c, self.config.max_forecast_temp_c))
        return np.array([air, mass], dtype=float)

    def _initial_covariance(self) -> np.ndarray:
        return np.diag([0.25, 1.0]).astype(float)

    def _estimate_filtered_states(
        self, cache: _SequenceCache, params: RoomRC2StateParams
    ) -> tuple[np.ndarray, np.ndarray]:
        previous = self.params
        self.params = params
        try:
            state = self._initial_state_from_sequence(cache.frame, params)
            covariance = self._initial_covariance()
            filtered_states = np.zeros((len(cache.frame), STATE_DIM), dtype=float)
            filtered_covariances = np.zeros((len(cache.frame), STATE_DIM, STATE_DIM), dtype=float)
            filtered_states[0] = state
            filtered_covariances[0] = covariance
            for index in range(1, len(cache.frame)):
                state, covariance = self.filter_update(
                    state,
                    covariance,
                    cache.inputs[index - 1],
                    cache.measurements[index],
                )
                filtered_states[index] = state
                filtered_covariances[index] = covariance
            return filtered_states, filtered_covariances
        finally:
            self.params = previous

    def _objective_for_sequences(
        self,
        caches: list[_SequenceCache],
        params: RoomRC2StateParams,
        horizons: Sequence[int],
        horizon_weights: dict[int, float],
    ) -> float:
        penalties = self._physical_penalty(params)
        total_loss = penalties
        total_weight = 1e-9
        _, _, A_d, B_d = self.params_to_matrices(params)
        for cache in caches:
            filtered_states, _ = self._estimate_filtered_states(cache, params)
            length = len(cache.frame)
            for horizon in horizons:
                if length <= horizon:
                    continue
                weight_h = horizon_weights[horizon]
                for start in range(length - horizon):
                    state = filtered_states[start].copy()
                    for offset in range(horizon):
                        u = cache.inputs[start + offset]
                        state = A_d @ state + B_d @ u
                    error = float(state[0] - cache.measurements[start + horizon])
                    sample_weight = cache.sample_weights[start] * weight_h
                    total_loss += sample_weight * self._robust_error(error)
                    total_weight += sample_weight
                    if (
                        state[0] < self.config.min_forecast_temp_c
                        or state[0] > self.config.max_forecast_temp_c
                    ):
                        overflow = max(
                            state[0] - self.config.max_forecast_temp_c,
                            self.config.min_forecast_temp_c - state[0],
                            0.0,
                        )
                        total_loss += self.config.physical_penalty_weight * (overflow**2)
        return total_loss / total_weight

    def _physical_penalty(self, params: RoomRC2StateParams) -> float:
        penalty = self.config.regularization_weight * (
            (params.eta_solar_air**2)
            + (params.eta_solar_mass**2)
            + (params.b_hour_sin_air**2)
            + (params.b_hour_cos_air**2)
            + (params.b_hour_sin_mass**2)
            + (params.b_hour_cos_mass**2)
        )
        if params.C_mass <= params.C_air:
            penalty += self.config.physical_penalty_weight * ((params.C_air - params.C_mass + 1e-6) ** 2)
        _, _, A_d, _ = self.params_to_matrices(params)
        spectral_radius = max(abs(np.linalg.eigvals(A_d)))
        if spectral_radius >= 1.0:
            penalty += self.config.spectral_radius_penalty * ((spectral_radius - 1.0 + 1e-6) ** 2)
        return float(penalty)

    def _robust_error(self, error_c: float) -> float:
        abs_error = abs(error_c)
        if self.config.use_huber_loss:
            delta = self.config.huber_delta_c
            if abs_error <= delta:
                return 0.5 * (error_c**2)
            return delta * (abs_error - 0.5 * delta)
        return abs_error

    def _collect_errors(
        self,
        caches: list[_SequenceCache],
        params: RoomRC2StateParams,
        horizons: Sequence[int],
    ) -> dict[int, list[float]]:
        errors = {int(h): [] for h in horizons}
        _, _, A_d, B_d = self.params_to_matrices(params)
        for cache in caches:
            filtered_states, _ = self._estimate_filtered_states(cache, params)
            length = len(cache.frame)
            for horizon in horizons:
                if length <= horizon:
                    continue
                for start in range(length - horizon):
                    state = filtered_states[start].copy()
                    for offset in range(horizon):
                        state = A_d @ state + B_d @ cache.inputs[start + offset]
                    errors[horizon].append(float(state[0] - cache.measurements[start + horizon]))
        return errors

    def _collect_segment_errors(
        self,
        caches: list[_SequenceCache],
        params: RoomRC2StateParams,
        horizons: Sequence[int],
    ) -> dict[str, dict[int, list[float]]]:
        segment_errors = {
            name: {int(h): [] for h in horizons} for name in DEFAULT_SEGMENT_DESCRIPTIONS
        }
        _, _, A_d, B_d = self.params_to_matrices(params)
        for cache in caches:
            filtered_states, _ = self._estimate_filtered_states(cache, params)
            length = len(cache.frame)
            for horizon in horizons:
                if length <= horizon:
                    continue
                for start in range(length - horizon):
                    state = filtered_states[start].copy()
                    for offset in range(horizon):
                        state = A_d @ state + B_d @ cache.inputs[start + offset]
                    error = float(state[0] - cache.measurements[start + horizon])
                    for name, mask in cache.segment_masks.items():
                        if bool(mask[start]):
                            segment_errors[name][horizon].append(error)
        return segment_errors

    def _metric_dict(self, horizon: int, errors: list[float]) -> dict[str, Any]:
        if not errors:
            return {
                "horizon_steps": int(horizon),
                "horizon_minutes": int(horizon * self.config.interval_minutes),
                "sample_count": 0,
                "mae_c": None,
                "rmse_c": None,
                "bias_c": None,
                "p95_abs_error_c": None,
            }
        arr = np.asarray(errors, dtype=float)
        return {
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon * self.config.interval_minutes),
            "sample_count": int(arr.size),
            "mae_c": float(np.mean(np.abs(arr))),
            "rmse_c": float(np.sqrt(np.mean(arr**2))),
            "bias_c": float(np.mean(arr)),
            "p95_abs_error_c": float(np.quantile(np.abs(arr), 0.95)),
        }

    def _log_training_context(
        self, prepared: pd.DataFrame, caches: list[_SequenceCache]
    ) -> None:
        LOGGER.info(
            "Training on %s samples from %s to %s across %s sequences",
            len(prepared),
            prepared["timestamp"].iloc[0],
            prepared["timestamp"].iloc[-1],
            len(caches),
        )
        if not self.config.use_segment_weights:
            return
        if not caches:
            return
        all_weights = np.concatenate([cache.sample_weights for cache in caches])
        LOGGER.info("Average training weight: %.3f", float(np.mean(all_weights)))
        counts: dict[str, int] = {name: 0 for name in DEFAULT_SEGMENT_DESCRIPTIONS}
        for cache in caches:
            for name, mask in cache.segment_masks.items():
                counts[name] += int(np.sum(mask))
        LOGGER.info("Segment counts: %s", counts)
