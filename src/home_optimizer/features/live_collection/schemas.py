from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class LiveMeasurement:
    name: str
    timestamp: datetime
    value: Any
