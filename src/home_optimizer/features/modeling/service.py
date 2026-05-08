from __future__ import annotations

import math

import numpy as np

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.models import (
    HorizonMetric,
    RoomModelConfig,
    RoomModelValidationReport,
    TrainedLinearRoomModel,
    ValidationFoldResult,
)


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


def _build_metric(
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


def _max_lag(config: RoomModelConfig) -> int:
    return max(
        config.room_temperature_lags
        + config.outdoor_temperature_lags
        + config.thermal_output_lags
        + config.solar_gain_lags
        + config.occupied_flag_lags
    )


def _row_value(row: MpcDatasetRow, field_name: str) -> float | None:
    value = getattr(row, field_name)
    if value is None:
        return None
    return float(value)


def _feature_specs(config: RoomModelConfig) -> list[tuple[str, str, int]]:
    specs: list[tuple[str, str, int]] = []
    specs.extend(
        ("room_temperature_c", f"room_temperature_lag_{lag}", lag)
        for lag in config.room_temperature_lags
    )
    specs.extend(
        ("outdoor_temperature_c", f"outdoor_temperature_lag_{lag}", lag)
        for lag in config.outdoor_temperature_lags
    )
    specs.extend(
        ("thermal_output_estimate_kw", f"thermal_output_lag_{lag}", lag)
        for lag in config.thermal_output_lags
    )
    specs.extend(
        ("solar_gain_proxy_w_m2", f"solar_gain_lag_{lag}", lag)
        for lag in config.solar_gain_lags
    )
    specs.extend(
        ("occupied_flag", f"occupied_flag_lag_{lag}", lag)
        for lag in config.occupied_flag_lags
    )
    return specs


def _default_feature_value(field_name: str) -> float | None:
    if field_name in {
        "outdoor_temperature_c",
        "thermal_output_estimate_kw",
        "solar_gain_proxy_w_m2",
        "occupied_flag",
    }:
        return 0.0
    return None


def _build_training_matrix(
    rows: list[MpcDatasetRow],
    config: RoomModelConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    max_lag = _max_lag(config)
    feature_specs = _feature_specs(config)
    feature_names = [feature_name for _, feature_name, _ in feature_specs]

    x_rows: list[list[float]] = []
    y_values: list[float] = []

    for source_index in range(max_lag, len(rows) - 1):
        next_room_temperature = rows[source_index + 1].room_temperature_c
        if next_room_temperature is None:
            continue

        feature_values: list[float] = []
        valid = True
        for field_name, _, lag in feature_specs:
            lagged_index = source_index - lag
            value = _row_value(rows[lagged_index], field_name)
            if value is None:
                value = _default_feature_value(field_name)
            if value is None:
                valid = False
                break
            feature_values.append(value)

        if not valid:
            continue

        x_rows.append(feature_values)
        y_values.append(float(next_room_temperature))

    if not x_rows:
        return np.zeros((0, len(feature_names))), np.zeros((0,)), feature_names

    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), feature_names


def _solve_ridge_regression(
    x_matrix: np.ndarray,
    y_values: np.ndarray,
    *,
    ridge_alpha: float,
) -> tuple[float, list[float]]:
    if x_matrix.size == 0 or y_values.size == 0:
        raise ValueError("not enough valid rows to fit room model")

    design_matrix = np.column_stack([np.ones(len(x_matrix)), x_matrix])
    if ridge_alpha == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y_values, rcond=None)
        return float(coefficients[0]), [float(value) for value in coefficients[1:]]

    penalty = np.eye(design_matrix.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0

    coefficients = np.linalg.solve(
        design_matrix.T @ design_matrix + penalty,
        design_matrix.T @ y_values,
    )
    return float(coefficients[0]), [float(value) for value in coefficients[1:]]


class RoomModelingService:
    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomModelConfig | None = None,
    ) -> TrainedLinearRoomModel:
        config = config or RoomModelConfig()
        x_matrix, y_values, feature_names = _build_training_matrix(dataset.rows, config)
        intercept, coefficients = _solve_ridge_regression(
            x_matrix,
            y_values,
            ridge_alpha=config.ridge_alpha,
        )

        return TrainedLinearRoomModel(
            trained_from_utc=dataset.start_time_utc,
            trained_to_utc=dataset.end_time_utc,
            interval_minutes=dataset.interval_minutes,
            config=config,
            feature_names=feature_names,
            intercept=intercept,
            coefficients=coefficients,
            sample_count=len(y_values),
        )

    def predict_next_room_temperature(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        predicted_room_temperatures = predicted_room_temperatures or {}
        prediction_origin_index = (
            source_index if prediction_origin_index is None else prediction_origin_index
        )

        feature_values: list[float] = []
        coefficient_index = 0
        for field_name, _, lag in _feature_specs(model.config):
            lagged_index = source_index - lag

            if field_name == "room_temperature_c" and lagged_index > prediction_origin_index:
                value = predicted_room_temperatures.get(lagged_index)
            else:
                value = _row_value(rows[lagged_index], field_name)

            if value is None:
                value = _default_feature_value(field_name)
            if value is None:
                return None

            feature_values.append(value)
            coefficient_index += 1

        return model.intercept + sum(
            value * model.coefficients[index]
            for index, value in enumerate(feature_values)
        )

    def simulate_horizon(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")

        predictions: dict[int, float] = {}
        for step in range(1, horizon_steps + 1):
            source_index = start_index + step - 1
            target_index = start_index + step
            prediction = self.predict_next_room_temperature(
                model,
                rows,
                source_index=source_index,
                predicted_room_temperatures=predictions,
                prediction_origin_index=start_index,
            )
            if prediction is None:
                return []
            predictions[target_index] = prediction

        return [predictions[start_index + step] for step in range(1, horizon_steps + 1)]

    def rolling_validate_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomModelConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomModelConfig()
        rows = dataset.rows
        if len(rows) < config.min_train_rows + 2:
            raise ValueError("dataset is too small for rolling validation")

        folds: list[ValidationFoldResult] = []
        fold_start = config.min_train_rows

        while fold_start < len(rows) - 1:
            validate_end_exclusive = min(fold_start + config.validation_window_rows, len(rows))

            if config.training_window_rows is None:
                train_start_index = 0
            else:
                train_start_index = max(0, fold_start - config.training_window_rows)

            training_rows = rows[train_start_index:fold_start]
            if len(training_rows) < config.min_train_rows:
                break

            training_dataset = MpcDataset(
                interval_minutes=dataset.interval_minutes,
                start_time_utc=training_rows[0].timestamp_utc,
                end_time_utc=training_rows[-1].timestamp_utc,
                rows=training_rows,
            )
            model = self.fit_room_model(training_dataset, config=config)

            metrics: list[HorizonMetric] = []
            for horizon_steps in config.validation_horizons_steps:
                errors: list[float] = []
                for origin_index in range(fold_start - 1, validate_end_exclusive - horizon_steps):
                    actual_room_temperature = rows[origin_index + horizon_steps].room_temperature_c
                    if actual_room_temperature is None:
                        continue

                    simulated = self.simulate_horizon(
                        model,
                        rows,
                        start_index=origin_index,
                        horizon_steps=horizon_steps,
                    )
                    if len(simulated) != horizon_steps:
                        continue

                    errors.append(simulated[-1] - actual_room_temperature)

                metrics.append(
                    _build_metric(
                        errors=errors,
                        horizon_steps=horizon_steps,
                        interval_minutes=dataset.interval_minutes,
                    )
                )

            folds.append(
                ValidationFoldResult(
                    train_start_utc=training_rows[0].timestamp_utc,
                    train_end_utc=training_rows[-1].timestamp_utc,
                    validate_start_utc=rows[fold_start].timestamp_utc,
                    validate_end_utc=rows[validate_end_exclusive - 1].timestamp_utc,
                    training_sample_count=model.sample_count,
                    metrics=metrics,
                )
            )
            fold_start = validate_end_exclusive

        aggregate_metrics: list[HorizonMetric] = []
        for horizon_steps in config.validation_horizons_steps:
            horizon_fold_metrics = [
                metric
                for fold in folds
                for metric in fold.metrics
                if metric.horizon_steps == horizon_steps and metric.sample_count > 0
            ]
            total_samples = sum(metric.sample_count for metric in horizon_fold_metrics)
            if total_samples == 0:
                aggregate_metrics.append(
                    HorizonMetric(
                        horizon_steps=horizon_steps,
                        horizon_minutes=horizon_steps * dataset.interval_minutes,
                        sample_count=0,
                    )
                )
                continue

            def weighted_average(
                getter: str,
                metrics: list[HorizonMetric] = horizon_fold_metrics,
            ) -> float | None:
                weighted_values = [
                    (getattr(metric, getter), metric.sample_count)
                    for metric in metrics
                    if getattr(metric, getter) is not None
                ]
                if not weighted_values:
                    return None
                return sum(value * count for value, count in weighted_values) / sum(
                    count for _, count in weighted_values
                )

            aggregate_metrics.append(
                HorizonMetric(
                    horizon_steps=horizon_steps,
                    horizon_minutes=horizon_steps * dataset.interval_minutes,
                    sample_count=total_samples,
                    mae_c=weighted_average("mae_c"),
                    rmse_c=weighted_average("rmse_c"),
                    bias_c=weighted_average("bias_c"),
                    p95_abs_error_c=weighted_average("p95_abs_error_c"),
                )
            )

        return RoomModelValidationReport(
            interval_minutes=dataset.interval_minutes,
            config=config,
            folds=folds,
            aggregate_metrics=aggregate_metrics,
        )
