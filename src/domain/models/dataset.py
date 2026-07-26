from datetime import timedelta
from enum import StrEnum

from pydantic import BaseModel

from domain.models.config import SensorReference


class Aggregation(StrEnum):
    MEAN = "mean"
    COUNT = "count"
    LAST = "last"
    FIRST = "first"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    MEDIAN = "median"
    SPREAD = "spread"
    STDDEV = "stddev"


class FillMethod(StrEnum):
    NONE = "none"
    NULL = "null"
    NUMBER = "number"
    PREVIOUS = "previous"
    LINEAR = "linear"


class DataDefinition(BaseModel):
    name: str
    sensor: SensorReference


class TimeSeriesDefinition(DataDefinition):
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod = FillMethod.NONE


class ForecastDefinition(DataDefinition):
    horizon: timedelta = timedelta(hours=48)


class DatasetDefinition(BaseModel):
    specs: list[DataDefinition] = []
