from __future__ import annotations

from home_optimizer.domain.models import DomainModel


class IdentificationDataset(DomainModel):
    timestamps: list[str]
    feature_names: list[str]
    target_name: str
    features: list[list[float]]
    targets: list[float]


class IdentificationResult(DomainModel):
    model_name: str
    interval_minutes: int
    sample_count: int
    train_sample_count: int
    test_sample_count: int
    coefficients: dict[str, float]
    intercept: float
    train_rmse: float
    test_rmse: float
    test_rmse_recursive: float
    target_name: str
