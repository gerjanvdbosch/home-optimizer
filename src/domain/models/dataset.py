from dataclasses import dataclass
from typing import Literal

from pandas._typing import MergeHow
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
    target_interval: str | None = None


class AttributeSeriesDefinition(DataDefinition): ...


@dataclass(frozen=True)
class JoinDefinition:
    left: str
    right: str
    left_on: tuple[str, ...]
    right_on: tuple[str, ...]
    how: MergeHow = "left"


class DatasetDefinition(BaseModel):
    definitions: list[DataDefinition] = []
    joins: list[JoinDefinition]
