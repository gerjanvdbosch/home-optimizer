from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from scipy.linalg import expm
from scipy.optimize import minimize

from home_optimizer.domain.models import DomainModel
from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.models import (
    HorizonMetric,
    RoomModelValidationReport,
    SegmentValidationReport,
    ValidationConfig,
    ValidationFoldResult,
)

LOGGER = logging.getLogger(__name__)

MODEL_TYPE = "room_rc"
ROOM_RC_MODEL_KIND = MODEL_TYPE
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
RC_VALIDITY_COLUMN = "is_valid_for_room_rc_identification"
RC_EXCLUSION_REASONS_COLUMN = "room_rc_exclusion_reasons"
OBJECTIVE_FAILURE_LOSS = 1e12
DISABLED_MASS_OUTDOOR_RESISTANCE_OHM = 1e12
DEGRADED_MIN_USABLE_SEQUENCE_COUNT = 2
DEGRADED_DROPPED_SEQUENCE_FRACTION_THRESHOLD = 0.5


@dataclass
class PreprocessingDiagnostics:
    missing_counts_before: dict[str, int]
    missing_counts_after_interpolation: dict[str, int]
    zero_filled_counts: dict[str, int]
    fraction_filled: dict[str, float]
    sample_count_before_filtering: int
    sample_count_after_filtering: int
    warnings: list[str]


@dataclass
class TrainingSequenceDiagnostics:
    min_sequence_length: int
    total_valid_sequence_count: int
    usable_sequence_count: int
    dropped_short_sequence_count: int
    usable_sample_count: int


@dataclass
class RoomRC2StateParams:
    R_air_out: float = 5.0
    R_air_mass: float = 1.0
    R_mass_out: float = 20.0
    C_air: float = 0.5
    C_mass: float = 50.0
    eta_heat: float = 0.9
    eta_solar_air: float = 0.2
    eta_solar_mass: float = 0.0
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
    valid_row_mask: np.ndarray


@dataclass
class PreparedRoomRcData:
    timestamps_utc: list[datetime]
    frame: pd.DataFrame
    segment_masks: dict[str, np.ndarray]


class RoomRcConfig(ValidationConfig):
    model_kind: str = ROOM_RC_MODEL_KIND
    interval_minutes: int = Field(default=10, gt=0)
    glass_area_m2: float = Field(default=8.0, ge=0.0)
    g_glass: float = Field(default=0.50, ge=0.0)
    shutter_mode: Literal["open_percent", "closed_percent"] = "open_percent"
    alpha_solar: float = Field(default=0.85)
    alpha_heat: float = Field(default=0.70)
    use_segment_weights: bool = True
    use_huber_loss: bool = True
    huber_delta_c: float = Field(default=0.25, gt=0.0)
    kalman_process_noise_air: float = Field(default=0.02, ge=0.0)
    kalman_process_noise_mass: float = Field(default=0.005, ge=0.0)
    kalman_measurement_noise: float = Field(default=0.03, ge=0.0)
    max_forecast_temp_c: float = 45.0
    min_forecast_temp_c: float = -10.0
    local_timezone: str | None = None
    gap_factor: float = Field(default=1.5, gt=1.0)
    short_gap_interpolation_limit: int = Field(default=2, ge=0)
    max_irradiance_interpolation_gap_minutes: int = Field(default=30, ge=0)
    optimizer_maxiter: int = Field(default=300, gt=0)
    min_valid_train_rows: int = Field(default=96, gt=1)
    spectral_radius_penalty: float = Field(default=1e5, ge=0.0)
    physical_penalty_weight: float = Field(default=1e3, ge=0.0)
    regularization_weight: float = Field(default=1e-3, ge=0.0)
    train_weight_freezing: float = Field(default=2.0, ge=0.0)
    train_weight_heating_active: float = Field(default=1.0, ge=0.0)
    train_weight_sunny_shutters_open: float = Field(default=1.0, ge=0.0)
    train_weight_night: float = Field(default=0.5, ge=0.0)
    sunny_irradiance_threshold_wm2: float = Field(default=150.0, ge=0.0)
    heating_active_threshold_kw: float = Field(default=0.40, ge=0.0)
    shutters_open_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    shutters_closed_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    cold_outdoor_temp_threshold_c: float = 5.0
    freezing_outdoor_temp_threshold_c: float = 0.0
    mild_outdoor_temp_threshold_c: float = 12.0
    night_start_hour: int = Field(default=22, ge=0, le=23)
    night_end_hour: int = Field(default=6, ge=0, le=23)
    sunny_midday_start_hour: int = Field(default=11, ge=0, le=23)
    sunny_midday_end_hour: int = Field(default=16, ge=1, le=24)
    R_air_out_min: float = Field(default=0.1, gt=0.0)
    R_air_out_max: float = Field(default=100.0, gt=0.0)
    R_air_mass_min: float = Field(default=0.1, gt=0.0)
    R_air_mass_max: float = Field(default=100.0, gt=0.0)
    R_mass_out_min: float = Field(default=0.1, gt=0.0)
    R_mass_out_max: float = Field(default=100.0, gt=0.0)
    C_air_min: float = Field(default=0.05, gt=0.0)
    C_air_max: float = Field(default=10.0, gt=0.0)
    C_mass_min: float = Field(default=1.0, gt=0.0)
    C_mass_max: float = Field(default=500.0, gt=0.0)
    eta_heat_min: float = 0.1
    eta_heat_max: float = 1.5
    eta_solar_air_min: float = 0.0
    eta_solar_air_max: float = 1.5
    eta_solar_mass_min: float = 0.0
    eta_solar_mass_max: float = 1.5
    eta_internal_min: float = 0.0
    eta_internal_max: float = 0.2
    hour_coeff_min: float = -0.5
    hour_coeff_max: float = 0.5
    initial_mass_offset_min: float = -5.0
    initial_mass_offset_max: float = 5.0
    missing_heating_policy: str = Field(default="drop")
    heating_off_threshold_kw: float = Field(default=0.05, ge=0.0)
    warn_missing_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    warn_bound_hit_count: int = Field(default=2, ge=1)
    fit_mass_outdoor_resistance: bool = False
    fit_eta_solar_mass: bool = False
    fit_mass_hour_terms: bool = False
    mass_capacity_ratio_min: float = Field(default=3.0, ge=1.0)
    notes: str = Field(
        default="Physical 2-state RC room model with exact discretization and Kalman-filtered mass state."
    )

    @field_validator("alpha_solar", "alpha_heat")
    @classmethod
    def _clip_alpha(cls, value: float) -> float:
        return float(np.clip(value, 0.0, 0.99))

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0

    @property
    def disabled_mass_outdoor_resistance_ohm(self) -> float:
        return DISABLED_MASS_OUTDOOR_RESISTANCE_OHM

    def with_interval_minutes(self, interval_minutes: int) -> "RoomRcConfig":
        return self.model_copy(update={"interval_minutes": interval_minutes})

    def bounds(self) -> list[tuple[float, float]]:
        mass_outdoor_resistance_bounds = (
            (
                self.disabled_mass_outdoor_resistance_ohm,
                self.disabled_mass_outdoor_resistance_ohm,
            )
            if not self.fit_mass_outdoor_resistance
            else (self.R_mass_out_min, self.R_mass_out_max)
        )
        eta_solar_mass_bounds = (
            (0.0, 0.0)
            if not self.fit_eta_solar_mass
            else (self.eta_solar_mass_min, self.eta_solar_mass_max)
        )
        mass_hour_bounds = (
            (0.0, 0.0)
            if not self.fit_mass_hour_terms
            else (self.hour_coeff_min, self.hour_coeff_max)
        )
        return [
            (self.R_air_out_min, self.R_air_out_max),
            (self.R_air_mass_min, self.R_air_mass_max),
            mass_outdoor_resistance_bounds,
            (self.C_air_min, self.C_air_max),
            (self.C_mass_min, self.C_mass_max),
            (self.eta_heat_min, self.eta_heat_max),
            (self.eta_solar_air_min, self.eta_solar_air_max),
            eta_solar_mass_bounds,
            (self.eta_internal_min, self.eta_internal_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            (self.hour_coeff_min, self.hour_coeff_max),
            mass_hour_bounds,
            mass_hour_bounds,
            (self.initial_mass_offset_min, self.initial_mass_offset_max),
        ]


class RoomRcModel(DomainModel):
    model_kind: str = ROOM_RC_MODEL_KIND
    trained_from_utc: datetime
    trained_to_utc: datetime
    interval_minutes: int
    config: RoomRcConfig
    params: dict[str, float]
    sample_count: int
    training_metadata: dict[str, Any] = Field(default_factory=dict)
    feature_schema: dict[str, Any] = Field(default_factory=dict)
    notes: str = Field(
        default="Stored physical RC room model parameters and metadata."
    )


class RoomRC2StatePhysicalModel:
    """Physical 2-state RC grey-box model for room temperature forecasting."""

    def __init__(self, config: RoomRcConfig):
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
            "config": self.config.model_dump(),
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
        model = cls(RoomRcConfig(**payload["config"]))
        model.params = RoomRC2StateParams.from_dict(payload["params"])
        model.training_metadata = payload.get("training_metadata", {})
        model.feature_schema = payload.get("feature_schema", model.feature_schema)
        created = payload.get("created_at_utc")
        if created is not None:
            model.created_at_utc = datetime.fromisoformat(created)
        return model

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared, _ = self._prepare_features_with_diagnostics(
            df,
            require_room_temp="room_temp_c" in df.columns,
        )
        return prepared

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
        r_mass_out = (
            params.R_mass_out
            if self.config.fit_mass_outdoor_resistance
            else self.config.disabled_mass_outdoor_resistance_ohm
        )

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
                    0.0,  # heating_kw_eff niet direct naar lucht
                    params.eta_solar_air / c_air,
                    0.0,
                    params.eta_internal / c_air,
                    params.b_hour_sin_air / c_air,
                    params.b_hour_cos_air / c_air,
                ],
                [
                    (1.0 / r_mass_out) / c_mass,
                    params.eta_heat / c_mass,  # heating_kw_eff naar massa/vloer
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
        prepared, diagnostics = self._prepare_features_with_diagnostics(df, require_room_temp=True)
        valid_sequences = self._split_into_valid_rc_sequences(prepared)
        if len(prepared) < self.config.min_train_rows:
            raise ValueError(
                f"Training data has {len(prepared)} rows, fewer than min_train_rows={self.config.min_train_rows}"
            )

        horizon_list = self._normalize_horizons(horizons, include_144=include_144)
        effective_horizons = [
            horizon
            for horizon in horizon_list
            if any(len(sequence) > horizon for sequence in valid_sequences)
        ]
        if not effective_horizons:
            raise ValueError(
                "No training horizons are feasible for the available RC-valid sequences. "
                f"Requested horizons were {list(horizon_list)}."
            )
        sequence_diagnostics = self._training_sequence_diagnostics(
            valid_sequences=valid_sequences,
            horizons=effective_horizons,
        )
        sequences = [
            sequence
            for sequence in valid_sequences
            if len(sequence) >= sequence_diagnostics.min_sequence_length
        ]
        sample_count = int(sequence_diagnostics.usable_sample_count)
        if diagnostics.sample_count_after_filtering < self.config.min_valid_train_rows:
            raise ValueError(
                f"RC-valid training data has {diagnostics.sample_count_after_filtering} rows, "
                f"fewer than min_valid_train_rows={self.config.min_valid_train_rows}"
            )
        if not sequences:
            raise ValueError(
                "No RC-valid training sequences remain after enforcing the minimum sequence length "
                f"of {sequence_diagnostics.min_sequence_length} rows for horizons {list(effective_horizons)}"
            )
        if sample_count < self.config.min_valid_train_rows:
            raise ValueError(
                f"Usable RC training data has {sample_count} rows after dropping short sequences, "
                f"fewer than min_valid_train_rows={self.config.min_valid_train_rows}"
            )

        weights = self._horizon_weights(effective_horizons)
        caches = [self._build_sequence_cache(sequence) for sequence in sequences]
        self._log_training_context(prepared, caches, diagnostics, sequence_diagnostics)

        def objective(vector: np.ndarray) -> float:
            params = RoomRC2StateParams.from_vector(vector)
            loss = self._objective_for_sequences(caches, params, effective_horizons, weights)
            if not np.isfinite(loss):
                return OBJECTIVE_FAILURE_LOSS
            return float(loss)

        x0 = self.params.to_vector()
        if not self.config.fit_mass_outdoor_resistance:
            x0[2] = self.config.disabled_mass_outdoor_resistance_ohm
        initial_loss = float(objective(x0))
        if initial_loss >= OBJECTIVE_FAILURE_LOSS:
            raise ValueError(
                "RC training objective is non-finite at the initial parameters. "
                "This usually indicates fragmented sequences or remaining invalid inputs after RC filtering."
            )
        result = minimize(
            objective,
            x0=x0,
            bounds=self.config.bounds(),
            method="L-BFGS-B",
            options={"maxiter": self.config.optimizer_maxiter},
        )
        fitted_params = RoomRC2StateParams.from_vector(result.x)
        train_loss = float(objective(result.x))
        parameters_changed = not np.allclose(result.x, x0, rtol=1e-6, atol=1e-8)
        improved_from_initial = train_loss < (initial_loss - 1e-9)
        if train_loss >= OBJECTIVE_FAILURE_LOSS:
            raise ValueError(
                "RC optimizer returned a non-finite objective. "
                "The model was not stored because the fitted parameters are not trustworthy."
            )
        if not bool(result.success) and not (parameters_changed or improved_from_initial):
            raise ValueError(
                "RC optimizer failed without improving the initial solution: "
                f"{result.message}"
            )
        self.params = fitted_params
        fit_quality, fit_quality_reasons = self._fit_quality(
            optimizer_success=bool(result.success),
            train_loss=train_loss,
            params=self.params,
            sequence_diagnostics=sequence_diagnostics,
        )

        validation_metrics = None
        if validation_df is not None:
            validation_metrics = self.evaluate(validation_df, horizons=self._normalize_horizons(horizon_list, include_144=False))

        self.training_metadata = {
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_sample_count": sample_count,
            "train_sequence_count": len(sequences),
            "train_valid_sequence_count_before_length_filter": sequence_diagnostics.total_valid_sequence_count,
            "train_dropped_short_sequence_count": sequence_diagnostics.dropped_short_sequence_count,
            "train_min_sequence_length": sequence_diagnostics.min_sequence_length,
            "horizons": list(horizon_list),
            "effective_train_horizons": list(effective_horizons),
            "horizon_weights": {str(k): v for k, v in weights.items()},
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(getattr(result, "nit", 0)),
            "final_train_loss": train_loss,
            "fit_quality": fit_quality,
            "fit_quality_reasons": fit_quality_reasons,
            "validation_metrics": validation_metrics,
            "rc_diagnostics": self._training_diagnostics_summary(
                prepared=prepared,
                diagnostics=diagnostics,
                caches=caches,
                params=self.params,
                sequence_diagnostics=sequence_diagnostics,
            ),
        }
        LOGGER.info("Fitted RC parameters: %s", self.params.to_dict())
        LOGGER.info("Final train loss: %.6f", train_loss)
        for warning in self.training_metadata["rc_diagnostics"]["warnings"]:
            LOGGER.warning("RC diagnostic warning: %s", warning)
        if validation_metrics is not None:
            LOGGER.info("Validation metrics: %s", validation_metrics)
        return {
            "model_type": MODEL_TYPE,
            "train_sample_count": sample_count,
            "train_sequence_count": len(sequences),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "final_train_loss": train_loss,
            "fit_quality": fit_quality,
            "fit_quality_reasons": fit_quality_reasons,
            "params": self.params.to_dict(),
            "validation_metrics": validation_metrics,
            "rc_diagnostics": self.training_metadata["rc_diagnostics"],
        }

    def predict_one_step(
        self, df: pd.DataFrame, initial_state: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        prepared, _ = self._prepare_features_with_diagnostics(df, require_room_temp=True)
        sequences = self._split_into_valid_rc_sequences(prepared)
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
        prepared, _ = self._prepare_features_with_diagnostics(future_df, require_room_temp=False)
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
        return self._filter_update_with_matrices(
            state=state,
            covariance=covariance,
            u=u,
            measurement_room_temp_c=measurement_room_temp_c,
            A_d=A_d,
            B_d=B_d,
        )

    def _filter_update_with_matrices(
        self,
        *,
        state: np.ndarray,
        covariance: np.ndarray,
        u: np.ndarray,
        measurement_room_temp_c: Optional[float],
        A_d: np.ndarray,
        B_d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        prepared, diagnostics = self._prepare_features_with_diagnostics(df, require_room_temp=True)
        sequences = self._split_into_valid_rc_sequences(prepared)
        caches = [self._build_sequence_cache(sequence) for sequence in sequences]
        horizon_list = self._normalize_horizons(horizons, include_144=False)
        errors_by_horizon = self._collect_errors(caches, self.params, horizon_list)
        metrics = [self._metric_dict(h, errors_by_horizon[h]) for h in horizon_list]
        return {"aggregate_metrics": metrics, "rc_diagnostics": {"sample_count_before_filtering": diagnostics.sample_count_before_filtering, "sample_count_after_filtering": diagnostics.sample_count_after_filtering}}

    def evaluate_segments(
        self,
        df: pd.DataFrame,
        horizons: Sequence[int] = (1, 6, 36, 72, 144),
    ) -> dict[str, Any]:
        prepared, diagnostics = self._prepare_features_with_diagnostics(df, require_room_temp=True)
        sequences = self._split_into_valid_rc_sequences(prepared)
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
        return {"segment_metrics": reports, "rc_diagnostics": {"sample_count_before_filtering": diagnostics.sample_count_before_filtering, "sample_count_after_filtering": diagnostics.sample_count_after_filtering}}

    def _prepare_features_with_diagnostics(
        self,
        df: pd.DataFrame,
        require_room_temp: bool,
    ) -> tuple[pd.DataFrame, PreprocessingDiagnostics]:
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
        for column in essential_numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        missing_before = frame[essential_numeric].isna().sum().to_dict()
        if any(count > 0 for count in missing_before.values()):
            LOGGER.warning("Missing values before interpolation: %s", missing_before)

        interpolation_columns = [
            column
            for column in ["room_temp_c", "outdoor_temp_c", "irradiance_wm2", "shutter_position", "occupied_flag"]
            if column in frame.columns
        ]
        if interpolation_columns:
            frame.loc[:, interpolation_columns] = frame.loc[:, interpolation_columns].interpolate(
                method="linear",
                limit=self.config.short_gap_interpolation_limit,
                limit_direction="both",
            )

        if "irradiance_wm2" in frame.columns:
            max_gap_steps = max(1, self.config.max_irradiance_interpolation_gap_minutes // self.config.interval_minutes)
            frame["irradiance_wm2"] = (
                frame.set_index("timestamp")["irradiance_wm2"]
                .interpolate(
                    method="time",
                    limit=max_gap_steps,
                    limit_direction="both",
                    limit_area="inside",
                )
                .reset_index(drop=True)
            )

        explicit_off_mask = self._explicit_heating_off_mask(frame)
        zero_filled_counts = {column: 0 for column in essential_numeric}
        if "heating_kw" in frame.columns:
            heating_missing_mask = frame["heating_kw"].isna()
            if self.config.missing_heating_policy == "allow_zero_when_off":
                fill_mask = heating_missing_mask & explicit_off_mask
            else:
                fill_mask = heating_missing_mask & explicit_off_mask
            zero_filled_counts["heating_kw"] = int(fill_mask.sum())
            frame.loc[fill_mask, "heating_kw"] = 0.0

        non_heating_zero_fill_columns = [
            column
            for column in ["shutter_position", "occupied_flag"]
            if column in frame.columns
        ]
        for column in non_heating_zero_fill_columns:
            missing_mask = frame[column].isna()
            zero_filled_counts[column] = int(missing_mask.sum())
            if int(missing_mask.sum()) > 0:
                frame.loc[missing_mask, column] = 0.0

        if "irradiance_wm2" in frame.columns:
            irradiance_missing_mask = frame["irradiance_wm2"].isna()
            zero_filled_counts["irradiance_wm2"] = int(irradiance_missing_mask.sum())
            if int(irradiance_missing_mask.sum()) > 0:
                LOGGER.warning(
                    "Filling remaining irradiance gaps with 0.0 after bounded interpolation: %s",
                    int(irradiance_missing_mask.sum()),
                )
                frame.loc[irradiance_missing_mask, "irradiance_wm2"] = 0.0

        missing_after_interpolation = frame[essential_numeric].isna().sum().to_dict()
        non_heating_missing = {
            key: value
            for key, value in missing_after_interpolation.items()
            if key != "heating_kw"
        }
        if any(count > 0 for count in non_heating_missing.values()):
            LOGGER.warning(
                "Remaining non-heating missing values after interpolation/fill: %s",
                non_heating_missing,
            )

        strict_columns = [
            column for column in ["room_temp_c", "outdoor_temp_c"] if column in frame.columns
        ]
        missing_after = frame[strict_columns].isna().sum()
        if int(missing_after.sum()) > 0:
            raise ValueError(
                f"Missing required state/weather values remain after interpolation: {missing_after.to_dict()}"
            )

        frame["heating_kw"] = frame["heating_kw"].clip(lower=0.0)
        frame["irradiance_wm2"] = frame["irradiance_wm2"].clip(lower=0.0)
        frame["occupied_flag"] = frame["occupied_flag"].clip(lower=0.0, upper=1.0)
        if RC_VALIDITY_COLUMN not in frame.columns:
            rc_reasons = self._derive_rc_exclusion_reasons(frame, require_room_temp=require_room_temp)
            frame[RC_EXCLUSION_REASONS_COLUMN] = rc_reasons
            frame[RC_VALIDITY_COLUMN] = rc_reasons.map(lambda reasons: len(reasons) == 0)
        else:
            frame[RC_VALIDITY_COLUMN] = frame[RC_VALIDITY_COLUMN].fillna(False).astype(bool)
            if RC_EXCLUSION_REASONS_COLUMN not in frame.columns:
                frame[RC_EXCLUSION_REASONS_COLUMN] = [[] for _ in range(len(frame))]
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
        sample_count_after_filtering = int(frame[RC_VALIDITY_COLUMN].sum()) if require_room_temp else len(frame)
        fraction_filled = {
            column: (
                (max(0, diagnostics_before - missing_after_interpolation.get(column, 0)) / len(frame))
                if len(frame)
                else 0.0
            )
            for column, diagnostics_before in missing_before.items()
        }
        warnings: list[str] = []
        for column, fraction in fraction_filled.items():
            if fraction >= self.config.warn_missing_fraction:
                warnings.append(f"high_filled_fraction:{column}={fraction:.3f}")
        if sample_count_after_filtering < self.config.min_valid_train_rows and require_room_temp:
            warnings.append(
                f"low_valid_sample_count:{sample_count_after_filtering}<{self.config.min_valid_train_rows}"
            )
        return frame, PreprocessingDiagnostics(
            missing_counts_before={key: int(value) for key, value in missing_before.items()},
            missing_counts_after_interpolation={
                key: int(value) for key, value in missing_after_interpolation.items()
            },
            zero_filled_counts=zero_filled_counts,
            fraction_filled=fraction_filled,
            sample_count_before_filtering=len(frame),
            sample_count_after_filtering=sample_count_after_filtering,
            warnings=warnings,
        )

    def _shutter_factor_from_position(self, shutter_position: float) -> float:
        raw = float(shutter_position)
        if self.config.shutter_mode == "open_percent":
            factor = raw / 100.0
        else:
            factor = 1.0 - (raw / 100.0)
        return float(np.clip(factor, 0.0, 1.0))

    def _explicit_heating_off_mask(self, frame: pd.DataFrame) -> pd.Series:
        if "heating_semantically_off" in frame.columns:
            return frame["heating_semantically_off"].fillna(False).astype(bool)
        explicit_off = pd.Series(False, index=frame.index, dtype=bool)
        if "mode_off" in frame.columns:
            explicit_off = explicit_off | frame["mode_off"].fillna(0).astype(int).eq(1)
        if "hp_electric_power_kw" in frame.columns:
            explicit_off = explicit_off | (
                pd.to_numeric(frame["hp_electric_power_kw"], errors="coerce")
                .fillna(np.inf)
                .le(self.config.heating_off_threshold_kw)
            )
        return explicit_off

    def _derive_rc_exclusion_reasons(
        self,
        frame: pd.DataFrame,
        *,
        require_room_temp: bool,
    ) -> pd.Series:
        existing = frame.get("room_rc_exclusion_reasons")
        if existing is not None:
            return existing.map(lambda item: item if isinstance(item, list) else [])

        room_valid = frame.get("is_valid_for_room_rc_identification")
        if room_valid is not None:
            inferred = room_valid.fillna(False).astype(bool)
            return inferred.map(lambda ok: [] if ok else ["precomputed_room_rc_invalid"])

        explicit_off_mask = self._explicit_heating_off_mask(frame)
        base_reasons = frame.get("exclusion_reasons")
        base_reasons = (
            base_reasons.map(lambda item: item if isinstance(item, list) else [])
            if base_reasons is not None
            else pd.Series([[] for _ in range(len(frame))], index=frame.index)
        )

        derived: list[list[str]] = []
        for index, reasons in enumerate(base_reasons):
            row_reasons = list(reasons)
            if require_room_temp and pd.isna(frame.at[index, "room_temp_c"]):
                row_reasons.append("missing_room_temperature")
            if pd.isna(frame.at[index, "outdoor_temp_c"]):
                row_reasons.append("missing_outdoor_temperature")
            if "heating_kw" in frame.columns and pd.isna(frame.at[index, "heating_kw"]) and not bool(explicit_off_mask.iloc[index]):
                row_reasons.append("missing_heating_input")
            derived.append(list(dict.fromkeys(row_reasons)))
        return pd.Series(derived, index=frame.index)

    def _exp_filter(self, values: np.ndarray, alpha: float) -> np.ndarray:
        if len(values) == 0:
            return np.array([], dtype=float)
        filtered = np.zeros_like(values, dtype=float)
        filtered[0] = 0.0 if not np.isfinite(values[0]) else float(values[0])
        for index in range(1, len(values)):
            current = filtered[index - 1] if not np.isfinite(values[index]) else float(values[index])
            filtered[index] = (alpha * filtered[index - 1]) + ((1.0 - alpha) * current)
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
        segment_masks = self.segment_masks(sequence)
        weights = self._training_weights(segment_masks, len(sequence))
        return _SequenceCache(
            frame=sequence,
            inputs=sequence.loc[:, INPUT_COLUMNS].to_numpy(dtype=float),
            measurements=sequence["room_temp_c"].to_numpy(dtype=float),
            sample_weights=weights,
            segment_masks=segment_masks,
            valid_row_mask=sequence[RC_VALIDITY_COLUMN].to_numpy(dtype=bool),
        )

    def _split_into_valid_rc_sequences(self, prepared: pd.DataFrame) -> list[pd.DataFrame]:
        gap_sequences = self.split_into_sequences(prepared)
        valid_sequences: list[pd.DataFrame] = []
        for sequence in gap_sequences:
            valid_mask = sequence[RC_VALIDITY_COLUMN].to_numpy(dtype=bool)
            start_index: int | None = None
            for index, is_valid in enumerate(valid_mask):
                if is_valid and start_index is None:
                    start_index = index
                if (not is_valid or index == len(valid_mask) - 1) and start_index is not None:
                    end_index = index if not is_valid else index + 1
                    candidate = self._recompute_sequence_filtered_inputs(
                        sequence.iloc[start_index:end_index].reset_index(drop=True)
                    )
                    if not candidate.empty:
                        valid_sequences.append(candidate)
                    start_index = None
        return valid_sequences

    def _recompute_sequence_filtered_inputs(self, sequence: pd.DataFrame) -> pd.DataFrame:
        if sequence.empty:
            return sequence
        result = sequence.copy()
        result["heating_kw_eff"] = self._exp_filter(
            result["heating_kw"].to_numpy(dtype=float),
            self.config.alpha_heat,
        )
        result["solar_glass_kw_filtered"] = self._exp_filter(
            result["solar_glass_kw"].to_numpy(dtype=float),
            self.config.alpha_solar,
        )
        return result

    def _training_sequence_diagnostics(
        self,
        *,
        valid_sequences: list[pd.DataFrame],
        horizons: Sequence[int],
    ) -> TrainingSequenceDiagnostics:
        min_sequence_length = max(2, max(int(horizon) for horizon in horizons) + 1)
        usable_sequences = [
            sequence for sequence in valid_sequences if len(sequence) >= min_sequence_length
        ]
        return TrainingSequenceDiagnostics(
            min_sequence_length=min_sequence_length,
            total_valid_sequence_count=len(valid_sequences),
            usable_sequence_count=len(usable_sequences),
            dropped_short_sequence_count=len(valid_sequences) - len(usable_sequences),
            usable_sample_count=int(sum(len(sequence) for sequence in usable_sequences)),
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

    def segment_masks(self, sequence: pd.DataFrame) -> dict[str, np.ndarray]:
        return self._segment_masks(sequence)

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
            _, _, A_d, B_d = self.params_to_matrices(params)
            state = self._initial_state_from_sequence(cache.frame, params)
            covariance = self._initial_covariance()
            filtered_states = np.zeros((len(cache.frame), STATE_DIM), dtype=float)
            filtered_covariances = np.zeros((len(cache.frame), STATE_DIM, STATE_DIM), dtype=float)
            filtered_states[0] = state
            filtered_covariances[0] = covariance
            for index in range(1, len(cache.frame)):
                state, covariance = self._filter_update_with_matrices(
                    state=state,
                    covariance=covariance,
                    u=cache.inputs[index - 1],
                    measurement_room_temp_c=cache.measurements[index],
                    A_d=A_d,
                    B_d=B_d,
                )
                filtered_states[index] = state
                filtered_covariances[index] = covariance
            return filtered_states, filtered_covariances
        finally:
            self.params = previous

    def _rollout_room_predictions(
        self,
        filtered_states: np.ndarray,
        inputs: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        sample_count = len(filtered_states) - horizon
        if sample_count <= 0:
            return np.array([], dtype=float)
        states = filtered_states[:sample_count].copy()
        for offset in range(horizon):
            states = (states @ A_d.T) + (inputs[offset : offset + sample_count] @ B_d.T)
        return states[:, 0]

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
                predictions = self._rollout_room_predictions(
                    filtered_states=filtered_states,
                    inputs=cache.inputs,
                    A_d=A_d,
                    B_d=B_d,
                    horizon=horizon,
                )
                targets = cache.measurements[horizon:]
                errors = predictions - targets
                weight_h = horizon_weights[horizon]
                sample_weights = cache.sample_weights[: len(errors)] * weight_h
                robust_losses = np.array(
                    [self._robust_error(float(error)) for error in errors],
                    dtype=float,
                )
                total_loss += float(np.sum(sample_weights * robust_losses))
                total_weight += float(np.sum(sample_weights))
                above = predictions - self.config.max_forecast_temp_c
                below = self.config.min_forecast_temp_c - predictions
                overflow = np.maximum(np.maximum(above, below), 0.0)
                if np.any(overflow > 0.0):
                    total_loss += float(
                        self.config.physical_penalty_weight * np.sum(overflow**2)
                    )
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
        min_mass_capacity = self.config.mass_capacity_ratio_min * params.C_air
        if params.C_mass < min_mass_capacity:
            penalty += self.config.physical_penalty_weight * (
                (min_mass_capacity - params.C_mass + 1e-6) ** 2
            )
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
                predictions = self._rollout_room_predictions(
                    filtered_states=filtered_states,
                    inputs=cache.inputs,
                    A_d=A_d,
                    B_d=B_d,
                    horizon=horizon,
                )
                horizon_errors = predictions - cache.measurements[horizon:]
                errors[horizon].extend(float(error) for error in horizon_errors)
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
                predictions = self._rollout_room_predictions(
                    filtered_states=filtered_states,
                    inputs=cache.inputs,
                    A_d=A_d,
                    B_d=B_d,
                    horizon=horizon,
                )
                horizon_errors = predictions - cache.measurements[horizon:]
                for name, mask in cache.segment_masks.items():
                    relevant_errors = horizon_errors[mask[: len(horizon_errors)]]
                    segment_errors[name][horizon].extend(
                        float(error) for error in relevant_errors
                    )
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
        self,
        prepared: pd.DataFrame,
        caches: list[_SequenceCache],
        diagnostics: PreprocessingDiagnostics,
        sequence_diagnostics: TrainingSequenceDiagnostics,
    ) -> None:
        LOGGER.info(
            "Training on %s samples from %s to %s across %s sequences",
            len(prepared),
            prepared["timestamp"].iloc[0],
            prepared["timestamp"].iloc[-1],
            len(caches),
        )
        LOGGER.info(
            "RC-valid samples after filtering: %s / %s",
            diagnostics.sample_count_after_filtering,
            diagnostics.sample_count_before_filtering,
        )
        LOGGER.info(
            "Usable RC training sequences: %s / %s (dropped short sequences: %s, min length: %s rows)",
            sequence_diagnostics.usable_sequence_count,
            sequence_diagnostics.total_valid_sequence_count,
            sequence_diagnostics.dropped_short_sequence_count,
            sequence_diagnostics.min_sequence_length,
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

    def _parameters_at_bounds(self, params: RoomRC2StateParams) -> list[str]:
        hits: list[str] = []
        vector = params.to_vector()
        names = list(params.to_dict().keys())
        for name, value, (lower, upper) in zip(names, vector, self.config.bounds(), strict=False):
            if np.isclose(lower, upper):
                continue
            if np.isclose(value, lower):
                hits.append(f"{name}@min")
            elif np.isclose(value, upper):
                hits.append(f"{name}@max")
        return hits

    def _mass_capacity_ratio(self, params: RoomRC2StateParams) -> float:
        if params.C_air <= 0.0:
            return float("inf")
        return float(params.C_mass / params.C_air)

    def _dropped_sequence_fraction(
        self, sequence_diagnostics: TrainingSequenceDiagnostics
    ) -> float:
        total = sequence_diagnostics.total_valid_sequence_count
        if total <= 0:
            return 0.0
        return float(sequence_diagnostics.dropped_short_sequence_count / total)

    def _training_diagnostics_summary(
        self,
        *,
        prepared: pd.DataFrame,
        diagnostics: PreprocessingDiagnostics,
        caches: list[_SequenceCache],
        params: RoomRC2StateParams,
        sequence_diagnostics: TrainingSequenceDiagnostics,
    ) -> dict[str, Any]:
        valid_counts: dict[str, int] = {name: 0 for name in DEFAULT_SEGMENT_DESCRIPTIONS}
        for cache in caches:
            for name, mask in cache.segment_masks.items():
                valid_counts[name] += int(np.sum(mask))
        warnings = list(diagnostics.warnings)
        bound_hits = self._parameters_at_bounds(params)
        mass_capacity_ratio = self._mass_capacity_ratio(params)
        dropped_sequence_fraction = self._dropped_sequence_fraction(sequence_diagnostics)
        if sequence_diagnostics.dropped_short_sequence_count > 0:
            warnings.append(
                "short_sequences_dropped:"
                f"{sequence_diagnostics.dropped_short_sequence_count}/"
                f"{sequence_diagnostics.total_valid_sequence_count}"
            )
        if len(bound_hits) >= self.config.warn_bound_hit_count:
            warnings.append(f"bound_hits:{','.join(bound_hits)}")
        if np.isclose(mass_capacity_ratio, self.config.mass_capacity_ratio_min):
            warnings.append(
                "weak_mass_air_separation:"
                f"{mass_capacity_ratio:.3f}"
            )
        return {
            "missing_counts_before": diagnostics.missing_counts_before,
            "missing_counts_after_interpolation": diagnostics.missing_counts_after_interpolation,
            "zero_filled_counts": diagnostics.zero_filled_counts,
            "fraction_filled": diagnostics.fraction_filled,
            "sample_count_before_filtering": diagnostics.sample_count_before_filtering,
            "sample_count_after_filtering": diagnostics.sample_count_after_filtering,
            "usable_sample_count": sequence_diagnostics.usable_sample_count,
            "min_training_sequence_length": sequence_diagnostics.min_sequence_length,
            "total_valid_sequence_count": sequence_diagnostics.total_valid_sequence_count,
            "usable_sequence_count": sequence_diagnostics.usable_sequence_count,
            "dropped_short_sequence_count": sequence_diagnostics.dropped_short_sequence_count,
            "dropped_sequence_count": sequence_diagnostics.dropped_short_sequence_count,
            "dropped_sequence_fraction": dropped_sequence_fraction,
            "segment_counts_after_rc_filtering": valid_counts,
            "parameter_values": params.to_dict(),
            "c_mass_air_ratio": mass_capacity_ratio,
            "parameters_at_bounds": bound_hits,
            "bound_hits_excluding_fixed_parameters": bound_hits,
            "warnings": warnings,
            "invalid_rc_reasons_top": (
                prepared.loc[~prepared[RC_VALIDITY_COLUMN], RC_EXCLUSION_REASONS_COLUMN]
                .explode()
                .value_counts()
                .head(10)
                .to_dict()
                if RC_EXCLUSION_REASONS_COLUMN in prepared.columns
                else {}
            ),
        }

    def _fit_quality(
        self,
        *,
        optimizer_success: bool,
        train_loss: float,
        params: RoomRC2StateParams,
        sequence_diagnostics: TrainingSequenceDiagnostics,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        bound_hits = self._parameters_at_bounds(params)
        dropped_sequence_fraction = self._dropped_sequence_fraction(sequence_diagnostics)
        if not optimizer_success:
            reasons.append("optimizer_not_converged")
        if len(bound_hits) >= self.config.warn_bound_hit_count:
            reasons.append("multiple_bound_hits")
        if (
            sequence_diagnostics.usable_sequence_count < DEGRADED_MIN_USABLE_SEQUENCE_COUNT
            or dropped_sequence_fraction > DEGRADED_DROPPED_SEQUENCE_FRACTION_THRESHOLD
        ):
            reasons.append("short_sequences_dropped")
        if train_loss > max(self.config.huber_delta_c, 0.1):
            reasons.append("elevated_train_loss")
        return ("good" if not reasons else "degraded"), reasons


class RoomRcTrainer:
    def _config_for_interval(self, config: RoomRcConfig, interval_minutes: int) -> RoomRcConfig:
        return config.with_interval_minutes(interval_minutes)

    def _dataset_frame(self, rows: list[MpcDatasetRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp_utc,
                    "room_temp_c": row.room_temperature_c,
                    "outdoor_temp_c": row.outdoor_temperature_c,
                    "heating_kw": row.space_heating_output_estimate_kw,
                    "irradiance_wm2": row.solar_irradiance_w_m2,
                    "shutter_position": row.shutter_position_pct,
                    "occupied_flag": row.occupied_flag,
                    RC_VALIDITY_COLUMN: row.is_valid_for_room_rc_identification,
                    RC_EXCLUSION_REASONS_COLUMN: row.room_rc_exclusion_reasons,
                    "mode_off": row.mode_off,
                    "hp_electric_power_kw": row.hp_electric_power_kw,
                }
                for row in rows
            ]
        )

    def _infer_interval_minutes(self, frame: pd.DataFrame, fallback_minutes: int) -> int:
        if len(frame) < 2:
            return fallback_minutes
        delta_minutes = (
            pd.to_datetime(frame.iloc[1]["timestamp"], utc=True)
            - pd.to_datetime(frame.iloc[0]["timestamp"], utc=True)
        ).total_seconds() / 60.0
        if delta_minutes <= 0:
            return fallback_minutes
        return int(round(delta_minutes))

    def max_history_rows(self, config: RoomRcConfig) -> int:
        return max(36, max(config.validation_horizons_steps, default=1))

    def validation_stride_rows(self, config: RoomRcConfig, interval_minutes: int) -> int:
        if config.validation_stride_rows is not None:
            return config.validation_stride_rows
        return max(1, 60 // interval_minutes)

    def prepare(self, rows: list[MpcDatasetRow], config: RoomRcConfig) -> PreparedRoomRcData:
        frame = self._dataset_frame(rows)
        inferred_interval_minutes = self._infer_interval_minutes(
            frame,
            fallback_minutes=config.interval_minutes,
        )
        physical = RoomRC2StatePhysicalModel(
            self._config_for_interval(config, inferred_interval_minutes)
        )
        prepared = physical.prepare_features(frame)
        segment_masks = physical.segment_masks(prepared)
        timestamps = [timestamp.to_pydatetime() for timestamp in prepared["timestamp"]]
        return PreparedRoomRcData(
            timestamps_utc=timestamps,
            frame=prepared,
            segment_masks=segment_masks,
        )

    def fit(self, dataset: MpcDataset, config: RoomRcConfig) -> RoomRcModel:
        prepared = self.prepare(dataset.rows, config)
        return self.fit_prepared(
            prepared,
            config=config,
            interval_minutes=dataset.interval_minutes,
            train_start_index=0,
            train_end_exclusive=len(prepared.frame),
        )

    def fit_prepared(
        self,
        prepared: PreparedRoomRcData,
        *,
        config: RoomRcConfig,
        interval_minutes: int,
        train_start_index: int = 0,
        train_end_exclusive: int | None = None,
    ) -> RoomRcModel:
        resolved_end = len(prepared.frame) if train_end_exclusive is None else train_end_exclusive
        train_frame = prepared.frame.iloc[train_start_index:resolved_end].reset_index(drop=True)
        resolved_config = self._config_for_interval(config, interval_minutes)
        physical = RoomRC2StatePhysicalModel(resolved_config)
        horizons = tuple(config.validation_horizons_steps)
        fit_report = physical.fit(
            train_frame,
            horizons=horizons,
            include_144=144 in horizons,
        )
        trained_to_index = max(train_start_index, resolved_end - 1)
        return RoomRcModel(
            trained_from_utc=prepared.timestamps_utc[train_start_index],
            trained_to_utc=prepared.timestamps_utc[trained_to_index],
            interval_minutes=interval_minutes,
            config=resolved_config,
            params=physical.get_params().to_dict(),
            sample_count=int(fit_report["train_sample_count"]),
            training_metadata=physical.training_metadata,
            feature_schema=physical.feature_schema,
        )

    def _physical_from_model(self, model: RoomRcModel) -> RoomRC2StatePhysicalModel:
        resolved_config = self._config_for_interval(model.config, model.interval_minutes)
        physical = RoomRC2StatePhysicalModel(resolved_config)
        physical.set_params(RoomRC2StateParams.from_dict(model.params))
        physical.training_metadata = dict(model.training_metadata)
        physical.feature_schema = dict(model.feature_schema)
        return physical

    def predict_next_prepared(
        self,
        model: RoomRcModel,
        prepared: PreparedRoomRcData,
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        simulated = self.simulate_horizon_prepared(
            model,
            prepared,
            start_index=source_index,
            horizon_steps=1,
        )
        if not simulated:
            return None
        return float(simulated[0])

    def predict_next(
        self,
        model: RoomRcModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        prepared = self.prepare(rows, model.config)
        return self.predict_next_prepared(
            model,
            prepared,
            source_index=source_index,
            predicted_room_temperatures=predicted_room_temperatures,
            prediction_origin_index=prediction_origin_index,
        )

    def estimate_current_state(
        self,
        model: RoomRcModel,
        rows: list[MpcDatasetRow],
    ) -> tuple[float, float]:
        prepared = self.prepare(rows, model.config)
        return self.estimate_current_state_prepared(model, prepared)

    def estimate_current_state_prepared(
        self,
        model: RoomRcModel,
        prepared: PreparedRoomRcData,
    ) -> tuple[float, float]:
        if prepared.frame.empty:
            raise ValueError("Cannot estimate current state from an empty prepared dataset")
        physical = self._physical_from_model(model)
        cache = physical._build_sequence_cache(prepared.frame)
        filtered_states, _ = physical._estimate_filtered_states(
            cache,
            RoomRC2StateParams.from_dict(model.params),
        )
        room_temp_c, mass_temp_c = filtered_states[-1]
        return float(room_temp_c), float(mass_temp_c)

    def simulate_horizon_prepared(
        self,
        model: RoomRcModel,
        prepared: PreparedRoomRcData,
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")
        if start_index < 0 or start_index >= len(prepared.frame) - 1:
            return []
        if start_index + horizon_steps >= len(prepared.frame):
            return []

        physical = self._physical_from_model(model)
        cache = physical._build_sequence_cache(prepared.frame)
        filtered_states, _ = physical._estimate_filtered_states(
            cache,
            RoomRC2StateParams.from_dict(model.params),
        )
        state = filtered_states[start_index].copy()
        _, _, A_d, B_d = physical.params_to_matrices(RoomRC2StateParams.from_dict(model.params))
        predictions: list[float] = []
        for step in range(1, horizon_steps + 1):
            input_index = start_index + step - 1
            state = A_d @ state + B_d @ cache.inputs[input_index]
            predictions.append(float(state[0]))
        return predictions

    def simulate_horizon(
        self,
        model: RoomRcModel,
        rows: list[MpcDatasetRow],
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        prepared = self.prepare(rows, model.config)
        return self.simulate_horizon_prepared(
            model,
            prepared,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )

    def segment_definitions(self, config: RoomRcConfig) -> list[tuple[str, str]]:
        return list(DEFAULT_SEGMENT_DESCRIPTIONS.items())


def room_rc_validation_report_from_metrics(
    *,
    model: RoomRcModel,
    metrics: dict[str, Any],
) -> RoomModelValidationReport:
    aggregate_metrics = [
        HorizonMetric(**metric) for metric in metrics.get("aggregate_metrics", [])
    ]
    segment_metrics = [
        SegmentValidationReport(
            segment_name=segment["segment_name"],
            description=segment["description"],
            metrics=[HorizonMetric(**metric) for metric in segment["metrics"]],
        )
        for segment in metrics.get("segment_metrics", [])
    ]
    return RoomModelValidationReport(
        interval_minutes=model.interval_minutes,
        config=model.config,
        folds=[
            ValidationFoldResult(
                train_start_utc=model.trained_from_utc,
                train_end_utc=model.trained_to_utc,
                validate_start_utc=model.trained_from_utc,
                validate_end_utc=model.trained_to_utc,
                training_sample_count=model.sample_count,
                metrics=aggregate_metrics,
            )
        ],
        aggregate_metrics=aggregate_metrics,
        segment_metrics=segment_metrics,
    )
