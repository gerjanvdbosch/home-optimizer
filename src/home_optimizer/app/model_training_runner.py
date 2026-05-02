from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Protocol

from home_optimizer.domain import IdentifiedModel
from home_optimizer.features.identification import MultiModelTrainer

LOGGER = logging.getLogger(__name__)


class TrainingWindowReader(Protocol):
    def sample_time_range(self) -> tuple[datetime | None, datetime | None]: ...


class FullDatasetModelTrainingRunner:
    def __init__(
        self,
        model_training_service: MultiModelTrainer,
        range_reader: TrainingWindowReader,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> None:
        self.model_training_service = model_training_service
        self.range_reader = range_reader
        self.interval_minutes = interval_minutes
        self.train_fraction = train_fraction

    def train_full_dataset_models(self) -> list[IdentifiedModel] | None:
        start_time, latest_sample_time = self.range_reader.sample_time_range()
        if start_time is None or latest_sample_time is None:
            LOGGER.info("Model training skipped: no telemetry samples available")
            return None

        end_time = latest_sample_time + timedelta(minutes=1)
        models = self.model_training_service.train_all_models(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=self.interval_minutes,
            train_fraction=self.train_fraction,
        )
        LOGGER.info(
            "Trained %s identified models on full dataset from %s until %s: %s",
            len(models),
            start_time.isoformat(),
            end_time.isoformat(),
            ", ".join(model.model_name for model in models),
        )
        return models
