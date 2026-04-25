from __future__ import annotations

from datetime import datetime
from typing import Any

from home_optimizer.domain.models import DomainModel


class LiveMeasurement(DomainModel):
    name: str
    timestamp: datetime
    value: Any
