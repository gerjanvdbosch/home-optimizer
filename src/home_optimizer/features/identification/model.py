from __future__ import annotations

import numpy as np

from .metrics import rmse
from .schemas import IdentificationDataset, IdentificationResult

class LinearModelIdentifier:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

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
        recursive_test_predictions = self.recursive_predict(
            dataset=dataset,
            coefficients=coefficients,
            start_index=split_index,
        )

        return IdentificationResult(
            model_name=self.model_name,
            interval_minutes=interval_minutes,
            sample_count=sample_count,
            train_sample_count=len(train_targets),
            test_sample_count=len(test_targets),
            coefficients=feature_coefficients,
            intercept=intercept,
            train_rmse=rmse(train_targets, train_predictions),
            test_rmse=rmse(test_targets, test_predictions),
            test_rmse_recursive=rmse(test_targets, recursive_test_predictions),
            target_name=dataset.target_name,
        )

    @staticmethod
    def recursive_predict(
        dataset: IdentificationDataset,
        coefficients: np.ndarray,
        *,
        start_index: int,
    ) -> np.ndarray:
        feature_index = (
            dataset.feature_names.index("previous_room_temperature")
            if "previous_room_temperature" in dataset.feature_names
            else None
        )
        features = np.asarray(dataset.features, dtype=float).copy()

        predictions: list[float] = []
        previous_prediction = float(dataset.targets[start_index - 1])

        for index in range(start_index, len(dataset.targets)):
            row = features[index].copy()
            if feature_index is not None:
                row[feature_index] = previous_prediction

            design_row = np.concatenate([[1.0], row])
            prediction = float(design_row @ coefficients)
            predictions.append(prediction)
            previous_prediction = prediction

        return np.asarray(predictions, dtype=float)
