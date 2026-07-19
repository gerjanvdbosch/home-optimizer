import ast
from datetime import datetime
from typing import Any

from domain.models import SolarForecastPoint


def parse_solar_forecast(point: dict[str, Any] | None) -> list[SolarForecastPoint]:
    if point is None:
        return []

    values = ast.literal_eval(point["value"])

    return [
        SolarForecastPoint(
            time=datetime.fromisoformat(timestamp),
            watts=float(watts),
        )
        for timestamp, watts in values.items()
    ]
