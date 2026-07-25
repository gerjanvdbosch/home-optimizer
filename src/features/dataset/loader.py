import ast
from datetime import datetime

import pandas as pd

from domain.models.config import ForecastSpec, TimeSeriesSpec
from domain.models.interface import DataLoader
from domain.time import parse_datetime


class TimeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, spec):
        return isinstance(spec, TimeSeriesSpec)

    def load(self, spec: TimeSeriesSpec, start: datetime, end: datetime) -> pd.DataFrame:
        sensor = self.resolver.resolve(spec.sensor)

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            aggregation=spec.aggregation,
            interval=spec.interval,
        )

        rows = [
            {
                "time": parse_datetime(point["time"]),
                spec.name: point["value"],
            }
            for point in points
            if point["value"] is not None
        ]

        df = pd.DataFrame(rows)

        # if spec.fill == "ffill":
        #     df[spec.name] = df[spec.name].ffill()
        #
        # elif spec.fill == "interpolate":
        #     df[spec.name] = df[spec.name].astype(float).interpolate()

        return df


class ForecastLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, spec):
        return isinstance(spec, ForecastSpec)

    def load(self, spec, start, end) -> pd.DataFrame:
        sensor = self.resolver.resolve(spec.sensor)

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
        )

        rows = []

        for point in points:
            values = ast.literal_eval(point["value"])

            for target_time, watts in values.items():
                rows.append(
                    {
                        "time": parse_datetime(target_time),
                        spec.name: float(watts),
                    }
                )

        return pd.DataFrame(rows)
