from __future__ import annotations

import numpy as np

from home_optimizer.domain import FLOOR_HEAT_STATE, HP_SUPPLY_TARGET_TEMPERATURE, THERMAL_OUTPUT

from ..metrics import rmse
from ..schemas import IdentificationDataset, IdentificationResult

MODEL_KIND = THERMAL_OUTPUT
MODEL_NAME = "linear_1step_thermal_output"
ACTIVE_TARGET_THRESHOLD = 0.05
ACTIVE_SCORE_THRESHOLD = 0.5
ACTIVE_INTERCEPT_KEY = "_active_intercept"
ACTIVE_TARGET_THRESHOLD_KEY = "_active_target_threshold"

THERMAL_OUTPUT_FEATURE_NAMES = [
    "previous_thermal_output",
    "previous_heating_demand",
    f"previous_{FLOOR_HEAT_STATE}",
    "outdoor_temperature",
    HP_SUPPLY_TARGET_TEMPERATURE,
]


def active_feature_name(feature_name: str) -> str:
    return f"active::{feature_name}"


def predict_thermal_output(
    *,
    coefficients: dict[str, float],
    intercept: float,
    previous_thermal_output: float,
    previous_heating_demand: float,
    previous_floor_heat_state: float,
    outdoor_temperature: float,
    supply_target_temperature: float,
) -> float:
    feature_values = {
        "previous_thermal_output": previous_thermal_output,
        "previous_heating_demand": previous_heating_demand,
        f"previous_{FLOOR_HEAT_STATE}": previous_floor_heat_state,
        "outdoor_temperature": outdoor_temperature,
        HP_SUPPLY_TARGET_TEMPERATURE: supply_target_temperature,
    }
    if not _predict_space_heating_active(coefficients, feature_values):
        return 0.0

    predicted_value = intercept + sum(
        coefficients[feature_name] * feature_values[feature_name]
        for feature_name in THERMAL_OUTPUT_FEATURE_NAMES
    )
    return max(0.0, float(predicted_value))


def _predict_space_heating_active(
    coefficients: dict[str, float],
    feature_values: dict[str, float],
) -> bool:
    if ACTIVE_INTERCEPT_KEY not in coefficients:
        return True

    active_score = coefficients[ACTIVE_INTERCEPT_KEY] + sum(
        coefficients.get(active_feature_name(feature_name), 0.0) * feature_values[feature_name]
        for feature_name in THERMAL_OUTPUT_FEATURE_NAMES
    )
    return active_score >= ACTIVE_SCORE_THRESHOLD


class ThermalOutputModelIdentifier:
    def identify(
        self,
        dataset: IdentificationDataset,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        if dataset.target_name != THERMAL_OUTPUT:
            raise ValueError(f"expected {THERMAL_OUTPUT} target dataset")

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

        active_targets = (targets > ACTIVE_TARGET_THRESHOLD).astype(float)
        train_active_targets = active_targets[:split_index]

        active_design = np.column_stack([np.ones(len(train_features)), train_features])
        active_coefficients_array, _, _, _ = np.linalg.lstsq(
            active_design,
            train_active_targets,
            rcond=None,
        )

        active_train_mask = train_targets > ACTIVE_TARGET_THRESHOLD
        magnitude_train_features = train_features[active_train_mask]
        magnitude_train_targets = train_targets[active_train_mask]
        if len(magnitude_train_targets) == 0:
            magnitude_train_features = train_features
            magnitude_train_targets = train_targets

        magnitude_design = np.column_stack(
            [np.ones(len(magnitude_train_features)), magnitude_train_features]
        )
        magnitude_coefficients_array, _, _, _ = np.linalg.lstsq(
            magnitude_design,
            magnitude_train_targets,
            rcond=None,
        )

        intercept = float(magnitude_coefficients_array[0])
        feature_coefficients = {
            name: float(value)
            for name, value in zip(
                THERMAL_OUTPUT_FEATURE_NAMES,
                magnitude_coefficients_array[1:],
                strict=True,
            )
        }
        feature_coefficients[ACTIVE_INTERCEPT_KEY] = float(active_coefficients_array[0])
        feature_coefficients[ACTIVE_TARGET_THRESHOLD_KEY] = ACTIVE_TARGET_THRESHOLD
        feature_coefficients.update(
            {
                active_feature_name(name): float(value)
                for name, value in zip(
                    THERMAL_OUTPUT_FEATURE_NAMES,
                    active_coefficients_array[1:],
                    strict=True,
                )
            }
        )

        train_predictions = self._predict_rows(
            feature_rows=train_features,
            coefficients=feature_coefficients,
            intercept=intercept,
        )
        test_predictions = self._predict_rows(
            feature_rows=test_features,
            coefficients=feature_coefficients,
            intercept=intercept,
        )
        recursive_test_predictions = self._recursive_predict(
            dataset=dataset,
            coefficients=feature_coefficients,
            intercept=intercept,
            start_index=split_index,
        )

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
            test_rmse_recursive=rmse(test_targets, recursive_test_predictions),
            target_name=dataset.target_name,
        )

    def _predict_rows(
        self,
        *,
        feature_rows: np.ndarray,
        coefficients: dict[str, float],
        intercept: float,
    ) -> np.ndarray:
        return np.asarray(
            [
                self._predict_row(
                    row=row,
                    coefficients=coefficients,
                    intercept=intercept,
                )
                for row in feature_rows
            ],
            dtype=float,
        )

    def _recursive_predict(
        self,
        *,
        dataset: IdentificationDataset,
        coefficients: dict[str, float],
        intercept: float,
        start_index: int,
    ) -> np.ndarray:
        feature_index = THERMAL_OUTPUT_FEATURE_NAMES.index("previous_thermal_output")
        features = np.asarray(dataset.features, dtype=float).copy()
        predictions: list[float] = []
        previous_prediction = float(dataset.targets[start_index - 1])

        for index in range(start_index, len(dataset.targets)):
            row = features[index].copy()
            row[feature_index] = previous_prediction
            prediction = self._predict_row(
                row=row,
                coefficients=coefficients,
                intercept=intercept,
            )
            predictions.append(prediction)
            previous_prediction = prediction

        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _predict_row(
        *,
        row: np.ndarray,
        coefficients: dict[str, float],
        intercept: float,
    ) -> float:
        return predict_thermal_output(
            coefficients=coefficients,
            intercept=intercept,
            previous_thermal_output=float(row[0]),
            previous_heating_demand=float(row[1]),
            previous_floor_heat_state=float(row[2]),
            outdoor_temperature=float(row[3]),
            supply_target_temperature=float(row[4]),
        )
