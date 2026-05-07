from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.features.identification import (
    IdentificationDataset,
    IdentificationDatasetRow,
    RecursiveRolloutEvaluationService,
    TemperaturePredictionContext,
)


class PersistencePredictor:
    def __init__(self, *, field_name: str) -> None:
        self.field_name = field_name

    def predict_next(self, context: TemperaturePredictionContext) -> float | None:
        if context.previous_predictions:
            return context.previous_predictions[-1]
        return getattr(context.current_row, self.field_name)


def _build_dataset() -> IdentificationDataset:
    start_time = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    room_values = [20.0, 20.2, 20.4, 20.6, 20.8, 21.0]
    dhw_values = [48.0, 47.8, 47.6, 47.4, 47.2, 47.0]

    rows = [
        IdentificationDatasetRow(
            timestamp_utc=start_time + timedelta(minutes=15 * index),
            room_temperature_c=room_values[index],
            dhw_top_temperature_c=dhw_values[index],
            room_target_min_temperature_c=20.0,
            room_target_max_temperature_c=20.9,
            dhw_target_min_temperature_c=47.1,
            dhw_target_max_temperature_c=50.0,
            mode_space=1,
            mode_dhw=0,
            mode_off=0,
            defrost_active=0,
            booster_heater_active=0,
            occupied_flag=1,
            dhw_draw_proxy_c=0.0,
            dhw_draw_detected=0,
            is_valid_for_room_identification=True,
            is_valid_for_dhw_identification=True,
            is_valid_for_cop_identification=False,
            exclusion_reasons=[],
        )
        for index in range(len(room_values))
    ]
    return IdentificationDataset(
        interval_minutes=15,
        start_time_utc=start_time,
        end_time_utc=start_time + timedelta(minutes=15 * len(rows)),
        rows=rows,
    )


def test_recursive_rollout_evaluation_service_computes_room_metrics() -> None:
    dataset = _build_dataset()
    service = RecursiveRolloutEvaluationService()

    evaluation = service.evaluate_room(
        dataset,
        predictor=PersistencePredictor(field_name="room_temperature_c"),
        horizon_hours=(1,),
    )

    room_one_step = evaluation.one_step_temperature_errors
    assert room_one_step.sample_count == 5
    assert room_one_step.mae == pytest.approx(0.2)
    assert room_one_step.rmse == pytest.approx(0.2)
    assert room_one_step.bias == pytest.approx(-0.2)
    assert room_one_step.max_absolute_error == pytest.approx(0.2)

    room_horizon = evaluation.horizons[0]
    assert room_horizon.horizon_label == "1h"
    assert room_horizon.horizon_steps == 4
    assert room_horizon.temperature_errors.sample_count == 8
    assert room_horizon.temperature_errors.mae == pytest.approx(0.5)
    assert room_horizon.temperature_errors.rmse == pytest.approx(
        (sum(value * value for value in [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8]) / 8) ** 0.5
    )
    assert room_horizon.temperature_errors.bias == pytest.approx(-0.5)
    assert room_horizon.predicted_comfort is not None
    assert room_horizon.predicted_comfort.overshoot_degree_hours == pytest.approx(0.0)
    assert room_horizon.actual_comfort is not None
    assert room_horizon.actual_comfort.overshoot_degree_hours == pytest.approx(0.025)
    assert room_horizon.actual_comfort.violation_minutes == pytest.approx(15.0)

def test_recursive_rollout_evaluation_service_computes_dhw_metrics() -> None:
    dataset = _build_dataset()
    service = RecursiveRolloutEvaluationService()

    evaluation = service.evaluate_dhw(
        dataset,
        predictor=PersistencePredictor(field_name="dhw_top_temperature_c"),
        horizon_hours=(1,),
    )

    dhw_one_step = evaluation.one_step_temperature_errors
    assert dhw_one_step.sample_count == 5
    assert dhw_one_step.mae == pytest.approx(0.2)
    assert dhw_one_step.bias == pytest.approx(0.2)

    dhw_horizon = evaluation.horizons[0]
    assert dhw_horizon.horizon_steps == 4
    assert dhw_horizon.temperature_errors.sample_count == 8
    assert dhw_horizon.temperature_errors.mae == pytest.approx(0.5)
    assert dhw_horizon.temperature_errors.bias == pytest.approx(0.5)
    assert dhw_horizon.predicted_comfort is not None
    assert dhw_horizon.predicted_comfort.undershoot_degree_hours == pytest.approx(0.0)
    assert dhw_horizon.actual_comfort is not None
    assert dhw_horizon.actual_comfort.undershoot_degree_hours == pytest.approx(0.025)
    assert dhw_horizon.actual_comfort.violation_minutes == pytest.approx(15.0)
