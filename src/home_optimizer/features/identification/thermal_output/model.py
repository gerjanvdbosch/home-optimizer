from __future__ import annotations

from home_optimizer.domain import FLOOR_HEAT_STATE, HP_SUPPLY_TARGET_TEMPERATURE, THERMAL_OUTPUT

from ..model import LinearModelIdentifier
from ..schemas import IdentificationDataset, IdentificationResult

MODEL_KIND = THERMAL_OUTPUT
MODEL_NAME = "linear_1step_thermal_output"

THERMAL_OUTPUT_FEATURE_NAMES = [
    "previous_thermal_output",
    "previous_heating_demand",
    f"previous_{FLOOR_HEAT_STATE}",
    "outdoor_temperature",
    HP_SUPPLY_TARGET_TEMPERATURE,
]


class ThermalOutputModelIdentifier:
    def __init__(self) -> None:
        self.identifier = LinearModelIdentifier(model_name=MODEL_NAME)

    def identify(
        self,
        dataset: IdentificationDataset,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        if dataset.target_name != THERMAL_OUTPUT:
            raise ValueError(f"expected {THERMAL_OUTPUT} target dataset")
        return self.identifier.identify(
            dataset,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
