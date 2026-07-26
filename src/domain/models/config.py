from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

HeatPumpMode = Literal[
    "heat",
    "cool",
]


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


class InfluxSensor(BaseModel):
    measurement: str
    entity_id: str
    field: str
    value_type: str | None = None


class SensorReference(BaseModel):
    entity_id: str = Field()
    attribute: str = Field()

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


class SolarForecastConfig(BaseModel):
    p10: SensorReference = Field(description="10e percentile")
    p50: SensorReference = Field(description="50e percentile")
    p90: SensorReference = Field(description="90e percentile")


class SolarConfig(BaseModel):
    production: SensorReference = Field()
    forecast: SolarForecastConfig = Field()


class BoilerConfig(BaseModel):
    top_temperature: SensorReference = Field()
    bottom_temperature: SensorReference = Field()


class HeatPumpConfig(BaseModel):
    status: SensorReference = Field()
    mode: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    boiler: BoilerConfig = Field()


class AppConfig(BaseModel):
    solar: SolarConfig = Field()
    heat_pump: HeatPumpConfig = Field()


class TrainRequest(BaseModel):
    days: int = Field(default=90)
