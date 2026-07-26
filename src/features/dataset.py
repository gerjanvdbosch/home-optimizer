import ast
from datetime import datetime

import pandas as pd

from domain.models.config import SensorReference
from domain.models.dataset import (
    DatasetDefinition,
    DataDefinition,
    Aggregation,
    FillMethod,
    TimeSeriesDefinition,
    ForecastDefinition,
)
from domain.models.interface import DataLoader
from domain.time import parse_datetime


class DatasetLoader:
    def __init__(self, loaders: list):
        self.loaders = loaders

    def load(self, dataset: DatasetDefinition, start: datetime, end: datetime) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for spec in dataset.specs:
            loader = self._find(spec)

            frame = loader.load(
                spec=spec,
                start=start,
                end=end,
            )

            frames.append(frame)

        return self._merge(frames)

    def _find(self, spec: DataDefinition):
        for loader in self.loaders:
            if loader.supports(spec):
                return loader

        raise ValueError(f"No loader found for sensor: {spec.name}")

    def _merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()

        result = frames[0]

        for frame in frames[1:]:
            result = result.merge(
                frame,
                on="time",
                how="outer",
            )

        return result.sort_values("time").reset_index(drop=True)


class DatasetBuilder:
    def __init__(self):
        self._specs: list[DataDefinition] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1min",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":
        self._specs.append(
            TimeSeriesDefinition(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
            )
        )

        return self

    def forecast(self, name: str, sensor: SensorReference) -> "DatasetBuilder":
        self._specs.append(
            ForecastDefinition(
                name=name,
                sensor=sensor,
            )
        )

        return self

    # def sensor(self, name: str, sensor: SensorReference):
    #     pass  # current value

    def build(self) -> DatasetDefinition:
        return DatasetDefinition(specs=self._specs)


class TimeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, spec):
        return isinstance(spec, TimeSeriesDefinition)

    def load(self, spec: TimeSeriesDefinition, start: datetime, end: datetime) -> pd.DataFrame:
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
        return isinstance(spec, ForecastDefinition)

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
