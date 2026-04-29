from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.models import DomainModel


class IdentifiedModel(DomainModel):
    model_kind: str
    model_name: str
    trained_at_utc: datetime
    training_start_time_utc: datetime
    training_end_time_utc: datetime
    interval_minutes: int
    sample_count: int
    train_sample_count: int
    test_sample_count: int
    coefficients: dict[str, float]
    intercept: float
    train_rmse: float
    test_rmse: float
    target_name: str
