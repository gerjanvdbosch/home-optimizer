from __future__ import annotations

from typing import Any

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import ensure_utc
from home_optimizer.domain.timeseries import SensorPoint
from home_optimizer.domain.units import parse_sensor_value


def map_history_points(
    history: list[dict[str, Any]],
    spec: SensorSpec,
) -> list[SensorPoint]:
    points: list[SensorPoint] = []

    for item in history:
        parsed = parse_sensor_value(item.get("state"), spec.unit)
        if parsed is None:
            continue

        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            parsed *= spec.conversion_factor

        ts_raw = item.get("last_changed") or item.get("last_updated")
        if not ts_raw:
            continue

        points.append(SensorPoint(timestamp=ensure_utc(ts_raw), value=parsed))

    return sorted(points, key=lambda point: point.timestamp)

