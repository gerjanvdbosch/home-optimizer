from datetime import datetime
from typing import Any

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
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


class InfluxSensor(BaseModel):
    measurement: str
    entity_id: str
    field: str


class InfluxPoint(BaseModel):
    time: datetime
    value: Any


class SolarForecastPoint(BaseModel):
    time: datetime
    watts: float


class SolarForecastState(BaseModel):
    p10: list[SolarForecastPoint]
    p50: list[SolarForecastPoint]
    p90: list[SolarForecastPoint]


class OptimizerState(BaseModel):
    updated: datetime
    solar_forecast: SolarForecastState


class SensorReferenceRequest(BaseModel):
    entity_id: str
    attribute: str

    @model_validator(mode="before")
    @classmethod
    def from_tuple(cls, value):
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
