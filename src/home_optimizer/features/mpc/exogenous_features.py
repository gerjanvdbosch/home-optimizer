from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from home_optimizer.domain.time import ensure_utc


def local_hour_sin_cos(
    timestamp_utc: datetime,
    *,
    local_timezone: str | None,
) -> tuple[float, float]:
    localized = ensure_utc(timestamp_utc)
    if local_timezone:
        localized = localized.astimezone(ZoneInfo(local_timezone))
    local_hour = localized.hour + (localized.minute / 60.0)
    angle = 2.0 * math.pi * local_hour / 24.0
    return float(math.sin(angle)), float(math.cos(angle))


def continue_exp_filter(
    values: list[float],
    *,
    alpha: float,
    initial_filtered_value: float,
) -> list[float]:
    if not values:
        return []
    filtered: list[float] = []
    previous = float(initial_filtered_value)
    for value in values:
        current = float(value)
        previous = (alpha * previous) + ((1.0 - alpha) * current)
        filtered.append(float(previous))
    return filtered


def trailing_exp_filter(
    values: list[float],
    *,
    alpha: float,
) -> float:
    if not values:
        return 0.0
    filtered = float(values[0])
    for value in values[1:]:
        filtered = (alpha * filtered) + ((1.0 - alpha) * float(value))
    return float(filtered)


def solar_gain_proxy_to_kw(
    solar_gain_proxy_w_m2: float,
    *,
    glass_area_m2: float,
    g_glass: float,
) -> float:
    return float(float(solar_gain_proxy_w_m2) * glass_area_m2 * g_glass / 1000.0)
