import pandas as pd

from dataclasses import dataclass
from datetime import datetime
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

    stable_pv: float = 0.0
    stable_wp: float = 0.0
    stable_grid: float = 0.0
    stable_load: float = 0.0

    forecast_df: pd.DataFrame | None = None

    solar_bias: float = 1.0
    load_bias: float = 1.0

    room_temp: float = 0.0

    dhw_top: float = 0.0
    dhw_bottom: float = 0.0
    dhw_setpoint: float = 0.0
