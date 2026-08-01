import ast
from datetime import datetime

import pandas as pd
from pandas._typing import MergeHow

from domain.models.config import SensorReference
from domain.models.dataset import (
    Aggregation,
    AttributeSeriesDefinition,
    AttributeTimeSeriesDefinition,
    DataDefinition,
    DatasetDefinition,
    FillMethod,
    JoinDefinition,
    TimeSeriesDefinition,
)
from domain.models.interface import DataLoader
from domain.time import parse_datetime


class DatasetLoader:
    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders

    def load(
        self,
        dataset: DatasetDefinition,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {}

        for definition in dataset.definitions:
            loader = self._find(definition)

            frames[definition.name] = loader.load(
                definition=definition,
                start=start,
                end=end,
            )

        return self._merge(
            frames=frames,
            joins=dataset.joins,
        )

    def _find(self, definition: DataDefinition) -> DataLoader:
        for loader in self.loaders:
            if loader.supports(definition):
                return loader

        raise ValueError(f"No loader found for sensor: {definition.name}")

    def _merge(
        self,
        frames: dict[str, pd.DataFrame],
        joins: list[JoinDefinition],
    ) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()

        if not joins:
            frames_iter = iter(frames.values())

            result = next(frames_iter).copy()

            for frame in frames_iter:
                result = result.merge(
                    frame,
                    on="time",
                    how="left",
                )

            return result.reset_index(drop=True)

        result = frames[joins[0].left].copy()
        used = {joins[0].left}

        for join in joins:
            if join.left not in used:
                raise ValueError(
                    f"Cannot join '{join.left}': it is not part of the current dataset"
                )

            if len(join.left_on) != len(join.right_on):
                raise ValueError(
                    "left_on and right_on must contain the same number of columns"
                )

            right = frames[join.right].copy()

            result = result.merge(
                right,
                left_on=list(join.left_on),
                right_on=list(join.right_on),
                how=join.how,
                suffixes=("", f"_{join.right}"),
            )

            for left_key, right_key in zip(
                join.left_on,
                join.right_on,
                strict=True,
            ):
                if left_key != right_key:
                    result = result.drop(
                        columns=f"{right_key}_{join.right}",
                        errors="ignore",
                    )

            used.add(join.right)

        return result.reset_index(drop=True)


class DatasetBuilder:
    def __init__(self):
        self._definitions: list[DataDefinition] = []
        self._joins: list[JoinDefinition] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1m",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":
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
        """
        Load a time series of attribute snapshots.

        Each row represents a forecast or snapshot created at `time` for a
        specific `target_time`. The `time` column indicates when the snapshot
        was created, while `target_time` indicates when the value applies.

        Example:
            time  | target_time  | p50
            ------|--------------|----
            10:00 | 10:00        | 300
            10:00 | 10:30        | 400
            10:00 | 11:00        | 500
            10:30 | 10:30        | 320
            10:30 | 11:00        | 450
        """

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

        self._definitions.append(
            AttributeSeriesDefinition(
                name=name,
                sensor=sensor,
            )
        )
        return self

    def join(
        self,
        left: str,
        right: str,
        on: tuple[str, ...] | None = None,
        left_on: tuple[str, ...] | None = None,
        right_on: tuple[str, ...] | None = None,
        how: MergeHow = "left",
    ) -> "DatasetBuilder":
        if on is not None:
            left_on = right_on = on

        if left_on is None or right_on is None:
            raise ValueError(
                "Either 'on' or both 'left_on' and 'right_on' must be provided"
            )

        if len(left_on) != len(right_on):
            raise ValueError("left_on and right_on must have the same length")

        self._joins.append(
            JoinDefinition(
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
                how=how,
            )
        )

        return self

    def build(self) -> DatasetDefinition:
        names = {definition.name for definition in self._definitions}

        for join in self._joins:
            if join.left not in names:
                raise ValueError(f"Unknown join source: {join.left}")

            if join.right not in names:
                raise ValueError(f"Unknown join source: {join.right}")

        return DatasetDefinition(
            definitions=self._definitions,
            joins=self._joins,
        )


class TimeSeriesLoader(DataLoader):
    def __init__(self, influx, resolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition):
        return isinstance(definition, TimeSeriesDefinition)

    def load(
        self, definition: TimeSeriesDefinition, start: datetime, end: datetime
    ) -> pd.DataFrame:
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
