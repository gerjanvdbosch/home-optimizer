from datetime import datetime

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


class TimeSeriesPoint(BaseModel):
    time: datetime
    value: float | None


class SensorReference(BaseModel):
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


class SolarForecast(BaseModel):
    p10: SensorReference
    p50: SensorReference
    p90: SensorReference


class UpdateRequest(BaseModel):
    solar_forecast: SolarForecast
