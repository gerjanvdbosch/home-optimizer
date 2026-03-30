import numpy as np
import pandas as pd
import tzlocal

from zoneinfo import ZoneInfo
from datetime import timezone


def round_half(x):
    return round(x * 2) / 2


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_kw(watts):
    try:
        value = float(watts)
        return value / 1000.0
    except (TypeError, ValueError):
        return 0.0


def add_cyclic_time_features(
    df: pd.DataFrame, col_name="timestamp", local_tz=True
) -> pd.DataFrame:
    """
    Voegt cyclische tijd-features toe.

    local_tz=True:  Gebruikt lokale tijd (geschikt voor Load/Base verbruik).
    local_tz=False: Gebruikt UTC (geschikt voor PV/Zon ivm voorkomen van zomertijd-sprongen).
    """
    if df is None or col_name not in df.columns:
        return df

    # 1. Converteer naar de juiste tijdzone
    if local_tz:
        tz = ZoneInfo(tzlocal.get_localzone_name())
    else:
        tz = timezone.utc

    df = df.copy()
    dt = df[col_name].dt.tz_convert(tz).dt

    # 2. Tijd van de dag (0..24 uur) met minuten precisie
    precise_hour = dt.hour + (dt.minute / 60.0)

    df["hour_sin"] = np.sin(2 * np.pi * precise_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * precise_hour / 24.0)

    # 2. Dag van de week (0..6, Maandag=0)
    df["day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7.0)

    # 3. Dag van het jaar (1..366) - Seizoenen
    df["doy_sin"] = np.sin(2 * np.pi * dt.dayofyear / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dayofyear / 366.0)

    return df
