import ast
from datetime import datetime

from domain.models import InfluxPoint, SolarForecastPoint


def parse_solar_forecast(raw: InfluxPoint | None) -> list[SolarForecastPoint]:
    if raw is None:
        return []

    values = ast.literal_eval(raw.value)

    return [
        SolarForecastPoint(
            time=datetime.fromisoformat(timestamp),
            watts=float(watts),
        )
        for timestamp, watts in values.items()
    ]
