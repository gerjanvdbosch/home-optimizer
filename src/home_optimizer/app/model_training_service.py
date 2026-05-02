from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import IdentifiedModel


class IdentificationTrainer(Protocol):
    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel: ...


class MultiModelTrainingService:
    def __init__(
        self,
        thermal_output_identification_service: IdentificationTrainer,
        room_temperature_identification_service: IdentificationTrainer,
    ) -> None:
        self.thermal_output_identification_service = thermal_output_identification_service
        self.room_temperature_identification_service = room_temperature_identification_service

    def train_all_models(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> list[IdentifiedModel]:
        thermal_output_model = self.thermal_output_identification_service.identify_and_store(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
        room_temperature_model = self.room_temperature_identification_service.identify_and_store(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
        return [thermal_output_model, room_temperature_model]
