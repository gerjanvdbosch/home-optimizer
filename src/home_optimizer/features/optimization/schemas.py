from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OptimizationWindow:
    start_time: datetime
    end_time: datetime
