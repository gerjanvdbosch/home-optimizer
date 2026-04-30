from __future__ import annotations

from datetime import date

from home_optimizer.domain.models import DomainModel


class RoomTemperatureBacktestDayResult(DomainModel):
    day: date
    overlap_count: int
    rmse: float | None
    bias: float | None
    max_absolute_error: float | None
    minimum_predicted_temperature: float | None
    maximum_predicted_temperature: float | None
    under_comfort_count: int | None = None
    over_comfort_count: int | None = None
    error: str | None = None


class RoomTemperatureBacktestResult(DomainModel):
    model_name: str
    interval_minutes: int
    start_date: date
    end_date: date
    total_days: int
    successful_days: int
    failed_days: int
    average_rmse: float | None
    average_bias: float | None
    average_max_absolute_error: float | None
    worst_day_by_rmse: date | None
    day_results: list[RoomTemperatureBacktestDayResult]
