from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.domain import BuildingTemperatureModel
from home_optimizer.infrastructure.database.building_model_repository import BuildingModelRepository
from home_optimizer.infrastructure.database.session import Database


def test_building_model_repository_returns_latest_model(tmp_path) -> None:
    database = Database(str(tmp_path / "models.sqlite"))
    database.init_schema()
    repository = BuildingModelRepository(database)

    first_model = BuildingTemperatureModel(
        model_name="linear_1step_room_temperature",
        trained_at_utc=datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc),
        training_start_time_utc=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        training_end_time_utc=datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        sample_count=100,
        train_sample_count=80,
        test_sample_count=20,
        coefficients={"previous_room_temperature": 0.9},
        intercept=0.1,
        train_rmse=0.05,
        test_rmse=0.1,
        target_name="room_temperature",
    )
    second_model = first_model.model_copy(
        update={
            "trained_at_utc": datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc),
            "coefficients": {"previous_room_temperature": 0.92},
            "test_rmse": 0.08,
        }
    )

    repository.save(first_model)
    repository.save(second_model)

    assert repository.latest() == second_model
