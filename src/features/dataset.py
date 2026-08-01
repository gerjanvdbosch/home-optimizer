import ast
from datetime import datetime
from functools import reduce

import pandas as pd

from domain.models.config import SensorReference
from domain.models.dataset import (
    Aggregation,
    AttributeSeriesDefinition,
    AttributeTimeSeriesDefinition,
    DataDefinition,
    DatasetDefinition,
    FillMethod,
    TimeSeriesDefinition,
)
from domain.models.interface import DataLoader
from domain.time import parse_datetime


class DatasetLoader:
    def __init__(self, loaders: list):
        self.loaders = loaders

    def load(
        self, dataset: DatasetDefinition, start: datetime, end: datetime
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for definition in dataset.definitions:
            loader = self._find(definition)

            frame = loader.load(
                definition=definition,
                start=start,
                end=end,
            )

            frames.append(frame)

        return self._merge(frames)

    def _find(self, definition: DataDefinition):
        for loader in self.loaders:
            if loader.supports(definition):
                return loader

        raise ValueError(f"No loader found for sensor: {definition.name}")

    def _merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [frame for frame in frames if not frame.empty]

        if not frames:
            return pd.DataFrame()

        result = reduce(
            lambda left, right: left.merge(
                right,
                on="time",
                how="outer",
            ),
            frames,
        )

        return result.sort_values("time").reset_index(drop=True)


class DatasetBuilder:
    def __init__(self):
        self._definitions: list[DataDefinition] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1m",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":
        self._definitions.append(
            TimeSeriesDefinition(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
            )
        )

        return self

    def attribute_timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1m",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":
        self._definitions.append(
            AttributeTimeSeriesDefinition(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
            )
        )

        return self

    def attribute_series(self, name: str, sensor: SensorReference):
        self._definitions.append(
            AttributeSeriesDefinition(
                name=name,
                sensor=sensor,
            )
        )
        return self

    def build(self) -> DatasetDefinition:
        return DatasetDefinition(definitions=self._definitions)


class TimeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition):
        return isinstance(definition, TimeSeriesDefinition)

    def load(
        self, definition: TimeSeriesDefinition, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """
        Load a regular time series.

        Each row represents a value at a specific point in time. The `time`
        column represents the timestamp at which the value applies.

        Example:
            time   | value
            -------|------
            10:00  | 300
            10:15  | 400
            10:30  | 500
        """

        sensor = self.resolver.resolve(definition.sensor)

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            aggregation=definition.aggregation,
            interval=definition.interval,
            fill=definition.fill,
        )

        rows = [
            {
                "time": parse_datetime(point["time"]),
                definition.name: point["value"],
            }
            for point in points
            if point["value"] is not None
        ]

        df = pd.DataFrame(rows)

        return df


class AttributeTimeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition):
        return isinstance(definition, AttributeTimeSeriesDefinition)

    def load(
        self, definition: AttributeTimeSeriesDefinition, start, end
    ) -> pd.DataFrame:
        """
        Load a time series of attribute snapshots.

        Each row represents a forecast or snapshot created at `time` for a
        specific `target_time`. The `time` column indicates when the snapshot
        was created, while `target_time` indicates when the value applies.

        Example:
            time  | target_time | p50
            ------|--------------|----
            10:00 | 10:00        | 300
            10:00 | 10:30        | 400
            10:00 | 11:00        | 500
            10:30 | 10:30        | 320
            10:30 | 11:00        | 450
        """
        sensor = self.resolver.resolve(definition.sensor)

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            aggregation=definition.aggregation,
            interval=definition.interval,
            fill=definition.fill,
        )

        rows = []

        for point in points:
            values = ast.literal_eval(point["value"])

            time = parse_datetime(point["time"])

            for target_time, value in values.items():
                rows.append(
                    {
                        "time": time,
                        "target_time": parse_datetime(target_time),
                        definition.name: float(value),
                    }
                )

        return pd.DataFrame(rows)


class AttributeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition):
        return isinstance(definition, AttributeSeriesDefinition)

    def load(self, definition: AttributeSeriesDefinition, start, end) -> pd.DataFrame:
        """
        Load a single attribute series.

        Each row represents a value at a specific point in time. The `time`
        column represents the timestamp at which the value applies.

        Example:
            time   | p50
            -------|----
            10:00  | 300
            10:30  | 400
            11:00  | 500
        """

        sensor = self.resolver.resolve(definition.sensor)

        point = self.influx.find(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
        )

        if point is None:
            return pd.DataFrame()

        rows = []

        values = ast.literal_eval(point["value"])

        for target_time, value in values.items():
            rows.append(
                {
                    "time": parse_datetime(target_time),
                    definition.name: float(value),
                }
            )

        return pd.DataFrame(rows)
