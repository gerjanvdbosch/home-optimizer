from __future__ import annotations

from home_optimizer.domain.models import DomainModel


class NumericPoint(DomainModel):
    timestamp: str
    value: float


class NumericSeries(DomainModel):
    name: str
    unit: str | None
    points: list[NumericPoint]


class TextPoint(DomainModel):
    timestamp: str
    value: str


class TextSeries(DomainModel):
    name: str
    points: list[TextPoint]


class RegressionMetrics(DomainModel):
    sample_count: int
    rmse: float
    mae: float
    r_squared: float


class TrainTestMetrics(DomainModel):
    train: RegressionMetrics
    test: RegressionMetrics


class RoomTemperatureModelResult(DomainModel):
    target_name: str
    input_names: list[str]
    sample_interval_minutes: int
    train_fraction: float
    coefficients: dict[str, float]
    metrics: TrainTestMetrics
    actual_series: NumericSeries
    predicted_series: NumericSeries
    residual_series: NumericSeries
