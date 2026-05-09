from __future__ import annotations

from collections.abc import Callable

from home_optimizer.features.dataset.models import MpcDataset
from home_optimizer.features.modeling.common.metrics import build_metric
from home_optimizer.features.modeling.models import (
    RoomModelConfig,
    RoomModelValidationReport,
    SegmentValidationReport,
    TrainedLinearRoomModel,
    ValidationFoldResult,
)


def rolling_validate_room_model(
    dataset: MpcDataset,
    *,
    config: RoomModelConfig,
    fit_model: Callable[[MpcDataset, RoomModelConfig], TrainedLinearRoomModel],
    simulate_horizon: Callable[[TrainedLinearRoomModel, list, int, int], list[float]],
    row_segments: Callable,
    segment_definitions: Callable[[RoomModelConfig], list[tuple[str, str]]],
    validation_stride_rows: Callable[[RoomModelConfig, int], int],
) -> RoomModelValidationReport:
    rows = dataset.rows
    if len(rows) < config.min_train_rows + 2:
        raise ValueError("dataset is too small for rolling validation")

    folds: list[ValidationFoldResult] = []
    all_errors_by_horizon: dict[int, list[float]] = {
        horizon_steps: [] for horizon_steps in config.validation_horizons_steps
    }
    segment_errors_by_horizon: dict[str, dict[int, list[float]]] = {
        segment_name: {
            horizon_steps: [] for horizon_steps in config.validation_horizons_steps
        }
        for segment_name, _ in segment_definitions(config)
    }
    fold_start = config.min_train_rows
    stride_rows = validation_stride_rows(config, dataset.interval_minutes)

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
        model = fit_model(training_dataset, config)

        metrics = []
        for horizon_steps in config.validation_horizons_steps:
            errors: list[float] = []
            for origin_index in range(fold_start - 1, validate_end_exclusive - horizon_steps):
                actual_room_temperature = rows[origin_index + horizon_steps].room_temperature_c
                if actual_room_temperature is None:
                    continue

                simulated = simulate_horizon(model, rows, origin_index, horizon_steps)
                if len(simulated) != horizon_steps:
                    continue

                error = simulated[-1] - actual_room_temperature
                errors.append(error)
                all_errors_by_horizon[horizon_steps].append(error)
                for segment_name in row_segments(rows[origin_index], config):
                    segment_errors_by_horizon[segment_name][horizon_steps].append(error)

            metrics.append(
                build_metric(
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
        fold_start += stride_rows

    aggregate_metrics = [
        build_metric(
            errors=all_errors_by_horizon[horizon_steps],
            horizon_steps=horizon_steps,
            interval_minutes=dataset.interval_minutes,
        )
        for horizon_steps in config.validation_horizons_steps
    ]
    segment_metrics = [
        SegmentValidationReport(
            segment_name=segment_name,
            description=description,
            metrics=[
                build_metric(
                    errors=segment_errors_by_horizon[segment_name][horizon_steps],
                    horizon_steps=horizon_steps,
                    interval_minutes=dataset.interval_minutes,
                )
                for horizon_steps in config.validation_horizons_steps
            ],
        )
        for segment_name, description in segment_definitions(config)
    ]

    return RoomModelValidationReport(
        interval_minutes=dataset.interval_minutes,
        config=config,
        folds=folds,
        aggregate_metrics=aggregate_metrics,
        segment_metrics=segment_metrics,
    )
