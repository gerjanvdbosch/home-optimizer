from __future__ import annotations

import numpy as np

from home_optimizer.features.system_identification.schemas import RegressionMetrics


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    total_variance = float(np.sum((actual - np.mean(actual)) ** 2))
    if total_variance == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - float(np.sum(residuals**2)) / total_variance

    return RegressionMetrics(
        sample_count=len(actual),
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
    )
