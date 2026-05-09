from __future__ import annotations

import math

from home_optimizer.features.modeling.models import HorizonMetric


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]

    fraction = rank - lower
    return sorted_values[lower] + ((sorted_values[upper] - sorted_values[lower]) * fraction)


def build_metric(
    *,
    errors: list[float],
    horizon_steps: int,
    interval_minutes: int,
) -> HorizonMetric:
    if not errors:
        return HorizonMetric(
            horizon_steps=horizon_steps,
            horizon_minutes=horizon_steps * interval_minutes,
            sample_count=0,
        )

    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    return HorizonMetric(
        horizon_steps=horizon_steps,
        horizon_minutes=horizon_steps * interval_minutes,
        sample_count=len(errors),
        mae_c=sum(absolute_errors) / len(absolute_errors),
        rmse_c=math.sqrt(sum(squared_errors) / len(squared_errors)),
        bias_c=sum(errors) / len(errors),
        p95_abs_error_c=_percentile(sorted(absolute_errors), 0.95),
    )
