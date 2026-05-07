from __future__ import annotations

from math import sqrt

from .models import DomainModel


class PredictionErrorMetrics(DomainModel):
    sample_count: int
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    max_absolute_error: float | None = None


class ComfortMetrics(DomainModel):
    sample_count: int
    undershoot_degree_hours: float = 0.0
    overshoot_degree_hours: float = 0.0
    violation_minutes: float = 0.0
    max_undershoot_c: float = 0.0
    max_overshoot_c: float = 0.0


class RecursiveRolloutHorizonMetrics(DomainModel):
    horizon_steps: int
    horizon_minutes: int
    horizon_label: str
    temperature_errors: PredictionErrorMetrics
    predicted_comfort: ComfortMetrics | None = None
    actual_comfort: ComfortMetrics | None = None


class TemperatureRolloutEvaluation(DomainModel):
    variable_name: str
    interval_minutes: int
    one_step_temperature_errors: PredictionErrorMetrics
    horizons: list[RecursiveRolloutHorizonMetrics]


def compute_prediction_error_metrics(
    actual_values: list[float],
    predicted_values: list[float],
) -> PredictionErrorMetrics:
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual_values and predicted_values must have equal length")

    if not actual_values:
        return PredictionErrorMetrics(sample_count=0)

    errors = [
        predicted - actual
        for actual, predicted in zip(actual_values, predicted_values, strict=True)
    ]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    sample_count = len(actual_values)
    return PredictionErrorMetrics(
        sample_count=sample_count,
        mae=sum(absolute_errors) / sample_count,
        rmse=sqrt(sum(squared_errors) / sample_count),
        bias=sum(errors) / sample_count,
        max_absolute_error=max(absolute_errors),
    )


def compute_comfort_metrics(
    predicted_values: list[float],
    minimum_values: list[float | None],
    maximum_values: list[float | None],
    *,
    interval_minutes: int,
) -> ComfortMetrics | None:
    if (
        len(predicted_values) != len(minimum_values)
        or len(predicted_values) != len(maximum_values)
    ):
        raise ValueError("comfort metric inputs must have equal length")

    valid_rows = [
        (predicted, minimum, maximum)
        for predicted, minimum, maximum in zip(
            predicted_values,
            minimum_values,
            maximum_values,
            strict=True,
        )
        if minimum is not None and maximum is not None
    ]
    if not valid_rows:
        return None

    interval_hours = interval_minutes / 60.0
    undershoot_degree_hours = 0.0
    overshoot_degree_hours = 0.0
    violation_minutes = 0.0
    max_undershoot_c = 0.0
    max_overshoot_c = 0.0

    for predicted, minimum, maximum in valid_rows:
        assert minimum is not None
        assert maximum is not None

        undershoot = max(0.0, minimum - predicted)
        overshoot = max(0.0, predicted - maximum)

        undershoot_degree_hours += undershoot * interval_hours
        overshoot_degree_hours += overshoot * interval_hours
        if undershoot > 0.0 or overshoot > 0.0:
            violation_minutes += interval_minutes
        max_undershoot_c = max(max_undershoot_c, undershoot)
        max_overshoot_c = max(max_overshoot_c, overshoot)

    return ComfortMetrics(
        sample_count=len(valid_rows),
        undershoot_degree_hours=undershoot_degree_hours,
        overshoot_degree_hours=overshoot_degree_hours,
        violation_minutes=violation_minutes,
        max_undershoot_c=max_undershoot_c,
        max_overshoot_c=max_overshoot_c,
    )
