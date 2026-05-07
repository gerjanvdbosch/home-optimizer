from __future__ import annotations

from dataclasses import dataclass

from .evaluation import RecursiveTemperaturePredictor, TemperaturePredictionContext


@dataclass(frozen=True, slots=True)
class PersistenceTemperaturePredictor(RecursiveTemperaturePredictor):
    field_name: str

    def predict_next(self, context: TemperaturePredictionContext) -> float | None:
        if context.previous_predictions:
            return context.previous_predictions[-1]
        return getattr(context.current_row, self.field_name)

