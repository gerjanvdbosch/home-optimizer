from __future__ import annotations

import math

from home_optimizer.domain import NumericSeries


def prediction_error_summary(
    *,
    predicted: NumericSeries,
    actual: NumericSeries,
) -> tuple[int, float | None, float | None, float | None]:
    actual_by_timestamp = {point.timestamp: point.value for point in actual.points}
    errors = [
        point.value - actual_by_timestamp[point.timestamp]
        for point in predicted.points
        if point.timestamp in actual_by_timestamp
    ]
    if not errors:
        return 0, None, None, None

    squared_errors = [error * error for error in errors]
    absolute_errors = [abs(error) for error in errors]
    overlap_count = len(errors)
    rmse = math.sqrt(sum(squared_errors) / overlap_count)
    bias = sum(errors) / overlap_count
    max_absolute_error = max(absolute_errors)
    return overlap_count, rmse, bias, max_absolute_error
