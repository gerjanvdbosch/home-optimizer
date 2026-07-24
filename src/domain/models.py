from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    influx_host: str = Field(
        default="homeassistant.local",
        description="InfluxDB host",
    )
    influx_port: int = Field(
        default=8086,
        description="InfluxDB port",
    )
    influx_username: str = Field(
        default="",
        description="InfluxDB username",
    )
    influx_password: str = Field(
        default="",
        description="InfluxDB password",
    )
    influx_database: str = Field(
        default="home_assistant",
        description="InfluxDB database",
    )
    data_path: Path = Field(
        default=Path("data"),
        description="Data path",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


class Storage(Protocol):
    def save(self, data: dict[str, Any]) -> None: ...

    def load(self) -> dict[str, Any]: ...


Aggregation = Literal[
    "mean",
    "last",
    "first",
    "min",
    "max",
    "sum",
    "median",
]


class Resample(BaseModel):
    interval: str
    aggregation: Aggregation


class InfluxSensor(BaseModel):
    measurement: str
    entity_id: str
    field: str
    value_type: str | None = None


class PowerPoint(BaseModel):
    time: datetime
    watts: float


class SolarForecastState(BaseModel):
    p10: list[PowerPoint] = Field(default_factory=list)
    p50: list[PowerPoint] = Field(default_factory=list)
    p90: list[PowerPoint] = Field(default_factory=list)

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class OptimizerState(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    solar_forecast: SolarForecastState = Field(default_factory=SolarForecastState)
    pv_production: list[PowerPoint] = Field(default_factory=list)


class SensorReferenceRequest(BaseModel):
    entity_id: str
    attribute: str

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attribute": "value",
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attribute": value[1],
            }

        return value


class SolarForecastRequest(BaseModel):
    p10: SensorReferenceRequest
    p50: SensorReferenceRequest
    p90: SensorReferenceRequest

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class UpdateRequest(BaseModel):
    solar_forecast: SolarForecastRequest
    pv_production: SensorReferenceRequest


class TrainRequest(BaseModel):
    solar_forecast: SolarForecastRequest
    pv_production: SensorReferenceRequest
    days: int = 90
