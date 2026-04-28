from __future__ import annotations

import numpy as np

from .metrics import rmse
from .schemas import IdentificationDataset, IdentificationResult

MODEL_NAME = "linear_1step_room_temperature"


class RoomTemperatureModelIdentifier:
    def identify(
        self,
        dataset: IdentificationDataset,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        features = np.asarray(dataset.features, dtype=float)
        targets = np.asarray(dataset.targets, dtype=float)

        sample_count = len(targets)
        split_index = max(1, min(sample_count - 1, int(sample_count * train_fraction)))

        train_features = features[:split_index]
        test_features = features[split_index:]
        train_targets = targets[:split_index]
        test_targets = targets[split_index:]

        train_design = np.column_stack([np.ones(len(train_features)), train_features])
        coefficients, _, _, _ = np.linalg.lstsq(train_design, train_targets, rcond=None)

        intercept = float(coefficients[0])
        feature_coefficients = {
            name: float(value)
            for name, value in zip(dataset.feature_names, coefficients[1:], strict=True)
        }

        train_predictions = train_design @ coefficients
        test_design = np.column_stack([np.ones(len(test_features)), test_features])
        test_predictions = test_design @ coefficients

        return IdentificationResult(
            model_name=MODEL_NAME,
            interval_minutes=interval_minutes,
            sample_count=sample_count,
            train_sample_count=len(train_targets),
            test_sample_count=len(test_targets),
            coefficients=feature_coefficients,
            intercept=intercept,
            train_rmse=rmse(train_targets, train_predictions),
            test_rmse=rmse(test_targets, test_predictions),
            target_name=dataset.target_name,
        )
