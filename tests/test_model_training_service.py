from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.domain import IdentifiedModel
from home_optimizer.features.identification import MultiModelTrainingService


class FakeIdentifiedModelTrainer:
    def __init__(self, model_kind: str, model_name: str) -> None:
        self.model_kind = model_kind
        self.model_name = model_name
        self.calls: list[tuple[str, str, int, float]] = []

    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel:
        self.calls.append(
            (start_time.isoformat(), end_time.isoformat(), interval_minutes, train_fraction)
        )
        return IdentifiedModel(
            model_kind=self.model_kind,
            model_name=self.model_name,
            trained_at_utc=datetime(2026, 4, 29, 2, 0, tzinfo=timezone.utc),
            training_start_time_utc=start_time,
            training_end_time_utc=end_time,
            interval_minutes=interval_minutes,
            sample_count=100,
            train_sample_count=80,
            test_sample_count=20,
            coefficients={},
            intercept=0.0,
            train_rmse=0.0,
            test_rmse=0.0,
            test_rmse_recursive=0.0,
            target_name=self.model_kind,
        )


def test_multi_model_training_service_trains_models_in_given_order() -> None:
    thermal_trainer = FakeIdentifiedModelTrainer(
        model_kind="thermal_output",
        model_name="linear_1step_thermal_output",
    )
    room_trainer = FakeIdentifiedModelTrainer(
        model_kind="room_temperature",
        model_name="linear_2state_room_temperature",
    )
    service = MultiModelTrainingService([thermal_trainer, room_trainer])
    start_time = datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 4, 29, 0, 0, tzinfo=timezone.utc)

    models = service.train_all_models(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=30,
        train_fraction=0.75,
    )

    assert [model.model_kind for model in models] == ["thermal_output", "room_temperature"]
    assert thermal_trainer.calls == [
        ("2026-04-20T00:00:00+00:00", "2026-04-29T00:00:00+00:00", 30, 0.75)
    ]
    assert room_trainer.calls == [
        ("2026-04-20T00:00:00+00:00", "2026-04-29T00:00:00+00:00", 30, 0.75)
    ]


def test_multi_model_training_service_requires_at_least_one_trainer() -> None:
    with pytest.raises(ValueError, match="trainers must not be empty"):
        MultiModelTrainingService([])

