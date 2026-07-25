import ast
from typing import Any

from domain.models.state import FloatPoint
from domain.time import parse_datetime


def parse_forecast(point: dict[str, Any] | None) -> list[FloatPoint]:
    if point is None:
        return []

    values = ast.literal_eval(point["value"])

    return [
        FloatPoint(
            time=parse_datetime(timestamp),
            value=float(value),
        )
        for timestamp, value in values.items()
    ]


def parse_timeseries(
    points: list[dict],
) -> list[FloatPoint]:
    return [
        FloatPoint(
            time=parse_datetime(point["time"]),
            value=float(point["value"]),
        )
        for point in points
        if point["value"] is not None
    ]
