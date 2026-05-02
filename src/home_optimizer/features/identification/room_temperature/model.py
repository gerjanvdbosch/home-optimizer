from __future__ import annotations

from home_optimizer.domain import FLOOR_HEAT_STATE, ROOM_TEMPERATURE

from ..model import LinearModelIdentifier
from ..schemas import IdentificationDataset, IdentificationResult

MODEL_KIND = "room_temperature"
MODEL_NAME = "linear_2state_room_temperature"
FLOOR_HEAT_STATE_FEATURE_NAME = FLOOR_HEAT_STATE


ROOM_TEMPERATURE_FEATURE_NAMES = [
    "previous_room_temperature",
    "outdoor_temperature",
    "gti_living_room_windows_adjusted",
    FLOOR_HEAT_STATE_FEATURE_NAME,
]


class RoomTemperatureModelIdentifier:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.identifier = LinearModelIdentifier(model_name=model_name)

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
