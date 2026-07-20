import ast
from typing import Any

from domain.models import PowerPoint
from domain.time import parse_datetime


def parse_solar_forecast(point: dict[str, Any] | None) -> list[PowerPoint]:
    if point is None:
        return []

    values = ast.literal_eval(point["value"])

    return [
        PowerPoint(
            time=parse_datetime(timestamp),
            watts=float(watts),
        )
        for timestamp, watts in values.items()
    ]


def parse_pv_production(
    points: list[dict],
) -> list[PowerPoint]:
    return [
        PowerPoint(
            time=parse_datetime(point["time"]),
            watts=float(point["value"]),
        )
        for point in points
        if point["value"] is not None
    ]
