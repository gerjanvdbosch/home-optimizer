import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum


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

    latitude: float = 0.0
    longitude: float = 0.0

    hvac_mode: int | None = None

    current_pv: float = 0.0
    current_wp: float = 0.0
    current_grid: float = 0.0

    stable_pv: float = 0.0
    stable_load: float = 0.0

    forecast_df: pd.DataFrame | None = None

    solar_bias: float = 1.0
    load_bias: float = 1.0

    pv_buffer: deque = field(default_factory=lambda: deque(maxlen=15))
    load_buffer: deque = field(default_factory=lambda: deque(maxlen=15))

    current_slot_start: datetime | None = None

    # Voor kWh berekeningen (houdt de stand van vorig kwartier bij)
    last_pv: float = None
    last_wp: float = None
    last_grid_import: float = None
    last_grid_export: float = None

    dhw_temp: float = 0.0
    dhw_setpoint: float = 0.0
