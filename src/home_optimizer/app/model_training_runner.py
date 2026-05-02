from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Protocol

from home_optimizer.domain import IdentifiedModel

LOGGER = logging.getLogger(__name__)


class TrainingWindowReader(Protocol):
    def sample_time_range(self) -> tuple[datetime | None, datetime | None]: ...


class IdentificationTrainer(Protocol):
    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel: ...


class FullDatasetModelTrainingRunner:
    def __init__(
        self,
        room_temperature_identification_service: IdentificationTrainer,
        range_reader: TrainingWindowReader,
        thermal_output_identification_service: IdentificationTrainer | None = None,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> None:
        self.room_temperature_identification_service = room_temperature_identification_service
        self.thermal_output_identification_service = thermal_output_identification_service
        self.range_reader = range_reader
        self.interval_minutes = interval_minutes
        self.train_fraction = train_fraction

    def train_full_dataset_model(self) -> IdentifiedModel | None:
        start_time, latest_sample_time = self.range_reader.sample_time_range()
        if start_time is None or latest_sample_time is None:
            LOGGER.info("Model training skipped: no telemetry samples available")
            return None

        end_time = latest_sample_time + timedelta(minutes=1)
        if self.thermal_output_identification_service is not None:
            thermal_model = self.thermal_output_identification_service.identify_and_store(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=self.interval_minutes,
                train_fraction=self.train_fraction,
            )
            LOGGER.info(
                "Trained thermal output model on full dataset from %s until %s: %s",
                start_time.isoformat(),
                end_time.isoformat(),
                thermal_model.model_name,
            )

        model = self.room_temperature_identification_service.identify_and_store(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=self.interval_minutes,
            train_fraction=self.train_fraction,
        )
        LOGGER.info(
            "Trained identified model on full dataset from %s until %s",
            start_time.isoformat(),
            end_time.isoformat(),
        )
        return model
