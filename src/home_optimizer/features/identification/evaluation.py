from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from home_optimizer.domain import (
    RecursiveRolloutHorizonMetrics,
    TemperatureRolloutEvaluation,
    compute_comfort_metrics,
    compute_prediction_error_metrics,
)
from home_optimizer.features.identification.models import (
    IdentificationDataset,
    IdentificationDatasetRow,
)


@dataclass(frozen=True, slots=True)
class TemperaturePredictionContext:
    current_row: IdentificationDatasetRow
    next_row: IdentificationDatasetRow
    rollout_rows: tuple[IdentificationDatasetRow, ...]
    step_index: int
    horizon_steps: int
    previous_predictions: tuple[float, ...]


class RecursiveTemperaturePredictor(Protocol):
    def predict_next(self, context: TemperaturePredictionContext) -> float | None: ...


@dataclass(frozen=True, slots=True)
class _TargetDefinition:
    name: str
    actual_field: str
    minimum_field: str
    maximum_field: str
    validity_field: str


_ROOM_TARGET = _TargetDefinition(
    name="room_temperature",
    actual_field="room_temperature_c",
    minimum_field="room_target_min_temperature_c",
    maximum_field="room_target_max_temperature_c",
    validity_field="is_valid_for_room_identification",
)
_DHW_TARGET = _TargetDefinition(
    name="dhw_top_temperature",
    actual_field="dhw_top_temperature_c",
    minimum_field="dhw_target_min_temperature_c",
    maximum_field="dhw_target_max_temperature_c",
    validity_field="is_valid_for_dhw_identification",
)


def _hours_to_steps(*, hours: int, interval_minutes: int) -> int:
    if hours <= 0:
        raise ValueError("hours must be greater than zero")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")

    total_minutes = hours * 60
    if total_minutes % interval_minutes != 0:
        raise ValueError("horizon must be divisible by dataset interval")
    return total_minutes // interval_minutes


class RecursiveRolloutEvaluationService:
    def evaluate_room(
        self,
        dataset: IdentificationDataset,
        *,
        predictor: RecursiveTemperaturePredictor,
        horizon_hours: Sequence[int] = (1, 3, 6, 12, 24),
    ) -> TemperatureRolloutEvaluation:
        return self._evaluate_target(
            dataset,
            predictor=predictor,
            target=_ROOM_TARGET,
            horizon_hours=horizon_hours,
        )

    def evaluate_dhw(
        self,
        dataset: IdentificationDataset,
        *,
        predictor: RecursiveTemperaturePredictor,
        horizon_hours: Sequence[int] = (1, 3, 6, 12, 24),
    ) -> TemperatureRolloutEvaluation:
        return self._evaluate_target(
            dataset,
            predictor=predictor,
            target=_DHW_TARGET,
            horizon_hours=horizon_hours,
        )

    def _evaluate_target(
        self,
        dataset: IdentificationDataset,
        *,
        predictor: RecursiveTemperaturePredictor,
        target: _TargetDefinition,
        horizon_hours: Sequence[int],
    ) -> TemperatureRolloutEvaluation:
        one_step = self._evaluate_rollout(
            dataset,
            predictor=predictor,
            target=target,
            horizon_steps=1,
            horizon_label="1-step",
        )
        horizons = [
            self._evaluate_rollout(
                dataset,
                predictor=predictor,
                target=target,
                horizon_steps=_hours_to_steps(
                    hours=hours,
                    interval_minutes=dataset.interval_minutes,
                ),
                horizon_label=f"{hours}h",
            )
            for hours in horizon_hours
        ]
        return TemperatureRolloutEvaluation(
            variable_name=target.name,
            interval_minutes=dataset.interval_minutes,
            one_step_temperature_errors=one_step.temperature_errors,
            horizons=horizons,
        )

    def _evaluate_rollout(
        self,
        dataset: IdentificationDataset,
        *,
        predictor: RecursiveTemperaturePredictor,
        target: _TargetDefinition,
        horizon_steps: int,
        horizon_label: str,
    ) -> RecursiveRolloutHorizonMetrics:
        actual_values: list[float] = []
        predicted_values: list[float] = []
        minimum_values: list[float | None] = []
        maximum_values: list[float | None] = []
        actual_minimum_values: list[float | None] = []
        actual_maximum_values: list[float | None] = []

        rows = dataset.rows
        for start_index in range(len(rows) - horizon_steps):
            start_row = rows[start_index]
            start_value = getattr(start_row, target.actual_field)
            if start_value is None or not getattr(start_row, target.validity_field):
                continue

            rollout_rows = rows[start_index : start_index + horizon_steps + 1]
            if any(not getattr(row, target.validity_field) for row in rollout_rows):
                continue

            predictions_for_rollout: list[float] = []
            rollout_actual_values: list[float] = []
            rollout_minimum_values: list[float | None] = []
            rollout_maximum_values: list[float | None] = []
            valid_rollout = True

            for step_offset in range(1, horizon_steps + 1):
                current_row = rollout_rows[step_offset - 1]
                next_row = rollout_rows[step_offset]
                next_actual = getattr(next_row, target.actual_field)
                if next_actual is None:
                    valid_rollout = False
                    break

                prediction = predictor.predict_next(
                    TemperaturePredictionContext(
                        current_row=current_row,
                        next_row=next_row,
                        rollout_rows=tuple(rollout_rows),
                        step_index=step_offset,
                        horizon_steps=horizon_steps,
                        previous_predictions=tuple(predictions_for_rollout),
                    )
                )
                if prediction is None:
                    valid_rollout = False
                    break

                predictions_for_rollout.append(prediction)
                rollout_actual_values.append(next_actual)
                rollout_minimum_values.append(getattr(next_row, target.minimum_field))
                rollout_maximum_values.append(getattr(next_row, target.maximum_field))

            if not valid_rollout:
                continue

            actual_values.extend(rollout_actual_values)
            predicted_values.extend(predictions_for_rollout)
            minimum_values.extend(rollout_minimum_values)
            maximum_values.extend(rollout_maximum_values)
            actual_minimum_values.extend(rollout_minimum_values)
            actual_maximum_values.extend(rollout_maximum_values)

        return RecursiveRolloutHorizonMetrics(
            horizon_steps=horizon_steps,
            horizon_minutes=horizon_steps * dataset.interval_minutes,
            horizon_label=horizon_label,
            temperature_errors=compute_prediction_error_metrics(actual_values, predicted_values),
            predicted_comfort=compute_comfort_metrics(
                predicted_values,
                minimum_values,
                maximum_values,
                interval_minutes=dataset.interval_minutes,
            ),
            actual_comfort=compute_comfort_metrics(
                actual_values,
                actual_minimum_values,
                actual_maximum_values,
                interval_minutes=dataset.interval_minutes,
            ),
        )
