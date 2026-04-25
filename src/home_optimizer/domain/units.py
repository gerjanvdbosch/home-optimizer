from __future__ import annotations

from typing import Any


def parse_sensor_value(value: Any, unit: str | None) -> Any:
    if isinstance(value, str):
        value = value.strip()

    if value in (None, "", "unknown", "unavailable", "none"):
        return None

    if unit == "bool":
        if value == "on":
            return True
        if value == "off":
            return False

    try:
        return float(value)
    except (ValueError, TypeError):
        return str(value)
