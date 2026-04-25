from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float


def parse_location(values: Mapping[str, object]) -> Location | None:
    latitude = _parse_coordinate(values.get("latitude"))
    longitude = _parse_coordinate(values.get("longitude"))
    if latitude is None or longitude is None:
        return None

    return Location(latitude=latitude, longitude=longitude)


def _parse_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
