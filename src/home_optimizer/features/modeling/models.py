from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from home_optimizer.domain.models import DomainModel


class RoomModelConfig(DomainModel):
    room_temperature_lags: list[int] = Field(default_factory=lambda: [0, 1])
    outdoor_temperature_lags: list[int] = Field(default_factory=lambda: [0])
    thermal_output_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6])
    solar_gain_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6, 12, 18])
    shutter_position_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6])
    solar_shutter_interaction_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6, 12])
    occupied_flag_lags: list[int] = Field(default_factory=lambda: [0])
    ridge_alpha: float = Field(default=0.0, ge=0.0)
    min_train_rows: int = Field(default=96, gt=1)
    training_window_rows: int | None = Field(default=None, gt=1)
    validation_window_rows: int = Field(default=144, gt=1)
    validation_stride_rows: int | None = Field(default=None, gt=0)
    validation_horizons_steps: list[int] = Field(default_factory=lambda: [1, 6, 36, 72, 144])
    sunny_irradiance_threshold_w_m2: float = Field(default=150.0, ge=0.0)
    heating_active_threshold_kw: float = Field(default=0.1, ge=0.0)
    shutters_open_min_pct: float = Field(default=75.0, ge=0.0, le=100.0)
    shutters_closed_max_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    sunny_midday_start_hour: int = Field(default=11, ge=0, le=23)
    sunny_midday_end_hour: int = Field(default=16, ge=1, le=24)

    @field_validator(
        "room_temperature_lags",
        "outdoor_temperature_lags",
        "thermal_output_lags",
        "solar_gain_lags",
        "shutter_position_lags",
        "solar_shutter_interaction_lags",
        "occupied_flag_lags",
        "validation_horizons_steps",
    )
    @classmethod
    def _validate_non_negative_int_list(cls, value: list[int]) -> list[int]:
        ordered = sorted(set(value))
        if not ordered:
            raise ValueError("lag/horizon lists cannot be empty")
        if ordered[0] < 0:
            raise ValueError("lag/horizon values must be non-negative")
        return ordered


class TrainedLinearRoomModel(DomainModel):
    trained_from_utc: datetime
    trained_to_utc: datetime
    interval_minutes: int
    config: RoomModelConfig
    feature_names: list[str]
    intercept: float
    coefficients: list[float]
    sample_count: int


class HorizonMetric(DomainModel):
    horizon_steps: int
    horizon_minutes: int
    sample_count: int
    mae_c: float | None = None
    rmse_c: float | None = None
    bias_c: float | None = None
    p95_abs_error_c: float | None = None


class ValidationFoldResult(DomainModel):
    train_start_utc: datetime
    train_end_utc: datetime
    validate_start_utc: datetime
    validate_end_utc: datetime
    training_sample_count: int
    metrics: list[HorizonMetric]


class RoomModelValidationReport(DomainModel):
    interval_minutes: int
    config: RoomModelConfig
    folds: list[ValidationFoldResult]
    aggregate_metrics: list[HorizonMetric]
    segment_metrics: list["SegmentValidationReport"] = Field(default_factory=list)


class SegmentValidationReport(DomainModel):
    segment_name: str
    description: str
    metrics: list[HorizonMetric]


LINEAR_ROOM_MODEL_TYPE = "linear_room"


class StoredRoomModelVersion(DomainModel):
    model_id: str
    model_type: str = LINEAR_ROOM_MODEL_TYPE
    created_at_utc: datetime
    is_active: bool = False
    model: TrainedLinearRoomModel
    validation_report: RoomModelValidationReport | None = None


class StoredRoomModelVersionSummary(DomainModel):
    model_id: str
    model_type: str = LINEAR_ROOM_MODEL_TYPE
    created_at_utc: datetime
    trained_from_utc: datetime
    trained_to_utc: datetime
    interval_minutes: int
    sample_count: int
    is_active: bool = False
    validation_mae_1h_c: float | None = None
    validation_mae_6h_c: float | None = None
    validation_mae_12h_c: float | None = None
    validation_mae_24h_c: float | None = None
    validation_bias_6h_c: float | None = None
    validation_p95_12h_c: float | None = None
