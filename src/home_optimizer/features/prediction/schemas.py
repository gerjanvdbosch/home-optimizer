from __future__ import annotations

from home_optimizer.domain import DomainModel, NumericSeries


class RoomTemperaturePrediction(DomainModel):
    model_name: str
    interval_minutes: int
    target_name: str
    room_temperature: NumericSeries
