from typing import Literal

from pydantic import BaseModel

from domain.models.config import SensorReference

Aggregation = Literal[
    "mean",
    "count",
    "last",
    "first",
    "min",
    "max",
    "sum",
    "median",
    "spread",
    "stddev",
]

FillMethod = Literal[
    "none",
    "null",
    "previous",
    "linear",
]


class DataDefinition(BaseModel):
    name: str
    sensor: SensorReference


class TimeSeriesDefinition(DataDefinition):
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod = "none"


class AttributeTimeSeriesDefinition(DataDefinition):
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod = "none"


class AttributeSeriesDefinition(DataDefinition): ...


class DatasetDefinition(BaseModel):
    definitions: list[DataDefinition] = []
