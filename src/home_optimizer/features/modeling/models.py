from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from home_optimizer.domain.models import DomainModel


class ValidationConfig(DomainModel):
    min_train_rows: int = Field(default=96, gt=1)
    training_window_rows: int | None = Field(default=None, gt=1)
    validation_window_rows: int = Field(default=144, gt=1)
    validation_stride_rows: int | None = Field(default=None, gt=0)
    validation_horizons_steps: list[int] = Field(default_factory=lambda: [1, 6, 36, 72, 144])

    @field_validator(
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
    config: ValidationConfig
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
    config: ValidationConfig
    folds: list[ValidationFoldResult]
    aggregate_metrics: list[HorizonMetric]
    segment_metrics: list["SegmentValidationReport"] = Field(default_factory=list)


class SegmentValidationReport(DomainModel):
    segment_name: str
    description: str
    metrics: list[HorizonMetric]

class StoredModelVersion(DomainModel):
    model_id: str
    model_type: str
    created_at_utc: datetime
    is_active: bool = False
    model: DomainModel
    validation_report: RoomModelValidationReport | None = None


class StoredModelVersionSummary(DomainModel):
    model_id: str
    model_type: str
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
