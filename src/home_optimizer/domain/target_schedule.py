from __future__ import annotations

from datetime import time

from pydantic import Field

from .models import DomainModel


class TemperatureTargetWindow(DomainModel):
    time: time
    target: float
    low_margin: float = Field(ge=0.0)
    high_margin: float = Field(ge=0.0)

    @property
    def minimum(self) -> float:
        return self.target - self.low_margin

    @property
    def maximum(self) -> float:
        return self.target + self.high_margin
