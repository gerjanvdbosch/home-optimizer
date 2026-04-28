from __future__ import annotations

from math import sqrt

import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sqrt(float(np.mean(np.square(actual - predicted))))
