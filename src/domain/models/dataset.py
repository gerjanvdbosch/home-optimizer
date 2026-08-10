from dataclasses import dataclass
from typing import Literal

from pandas._typing import MergeHow
from pydantic import BaseModel

from domain.models.config import SensorAttributesReference, SensorReference

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
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod | int | float = "none"


class TimeSeriesDefinition(DataDefinition):
    sensor: SensorReference


class AttributeSeriesDefinition(DataDefinition):
    sensor: SensorAttributesReference
    attributes: list[str]
    target_interval: str | None = None


class AttributeTimeSeriesDefinition(AttributeSeriesDefinition):
    time_column: str = "time"


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
