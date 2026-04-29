from __future__ import annotations

from home_optimizer.domain import ROOM_TEMPERATURE

from ..model import LinearModelIdentifier
from ..schemas import IdentificationDataset, IdentificationResult

MODEL_KIND = "room_temperature"
MODEL_NAME = "linear_1step_room_temperature"


class RoomTemperatureModelIdentifier:
    def __init__(self) -> None:
        self.identifier = LinearModelIdentifier(model_name=MODEL_NAME)

    def identify(
        self,
        dataset: IdentificationDataset,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        if dataset.target_name != ROOM_TEMPERATURE:
            raise ValueError(f"expected {ROOM_TEMPERATURE} target dataset")
        return self.identifier.identify(
            dataset,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
