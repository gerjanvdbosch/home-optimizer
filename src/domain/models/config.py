from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

HeatPumpMode = Literal[
    "heat",
    "cool",
]

ForecasterType = Literal[
    "solar",
    "boiler",
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
    attribute: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attribute": None,
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
    state: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    compressor_frequency: SensorReference = Field()
    boiler: BoilerConfig = Field()


class AppConfig(BaseModel):
    solar: SolarConfig = Field()
    heat_pump: HeatPumpConfig = Field()


class FitRequest(BaseModel):
    forecaster: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class BacktestRequest(BaseModel):
    forecaster: ForecasterType
    days: int = Field(default=90)


class TuneRequest(BacktestRequest):
    trails: int = Field(default=3)


class JobType(str, Enum):
    FIT = "fit"
    TUNE = "tune"
    BACKTEST = "backtest"


@dataclass
class Job:
    id: str
    type: JobType
    request: FitRequest | TuneRequest | BacktestRequest
