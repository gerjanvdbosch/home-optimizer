import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
from typing import Optional


class HvacMode(Enum):
    OFF = 0
    DHW = 1
    HEATING = 2
    COOLING = 3
    LEGIONELLA_PREVENTION = 4
    FROST_PROTECTION = 5


@dataclass
class Context:
    now: datetime

    hvac_mode: int | None = None

    current_pv: float = 0.0
    current_wp: float = 0.0
    current_grid: float = 0.0

    stable_pv: float = 0.0
    stable_load: float = 0.0

    forecast_df: pd.DataFrame | None = None

    solar_bias: float = 1.0
    load_bias: float = 1.0

    pv_buffer: deque = field(default_factory=lambda: deque(maxlen=2))
    load_buffer: deque = field(default_factory=lambda: deque(maxlen=2))

    current_slot_start: datetime | None = None

    pv_samples: list[float] = field(default_factory=list)
    wp_samples: list[float] = field(default_factory=list)
    grid_samples: list[float] = field(default_factory=list)

    dhw_temp: float = 0.0
    dhw_setpoint: float = 0.0
