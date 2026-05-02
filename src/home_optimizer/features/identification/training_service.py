from __future__ import annotations

from datetime import datetime
from typing import Sequence

from home_optimizer.domain import IdentifiedModel

from .ports import IdentifiedModelTrainer


class MultiModelTrainingService:
    def __init__(self, trainers: Sequence[IdentifiedModelTrainer]) -> None:
        if not trainers:
            raise ValueError("trainers must not be empty")
        self.trainers = tuple(trainers)

    def train_all_models(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> list[IdentifiedModel]:
        return [
            trainer.identify_and_store(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                train_fraction=train_fraction,
            )
            for trainer in self.trainers
        ]

