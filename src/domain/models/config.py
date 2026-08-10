from pathlib import Path
from typing import Generic, Literal, TypeVar

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


T = TypeVar("T", bound=BaseModel)


class SensorAttributesReference(BaseModel, Generic[T]):
    entity_id: str = Field()
    attributes: T

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attributes": value[1],
            }

        return value


class SolarConfig(BaseModel):
    production: SensorReference = Field()


class BoilerConfig(BaseModel):
    setpoint: SensorReference = Field()
    top_temperature: SensorReference = Field()
    bottom_temperature: SensorReference = Field()


class HeatPumpConfig(BaseModel):
    state: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    compressor_frequency: SensorReference = Field()
    boiler: BoilerConfig = Field()


class SolcastAttributes(BaseModel):
    p10: str = Field(description="10e percentile")
    p50: str = Field(description="50e percentile")
    p90: str = Field(description="90e percentile")


class SolcastConfig(SensorAttributesReference[SolcastAttributes]): ...


class OpenMeteoAttributes(BaseModel):
    temperature: str = Field()
    gti: str = Field()
    cloud_cover: str = Field()
    wind_direction: str = Field()
    wind_speed: str = Field()
    precipitation: str = Field()


class OpenMeteoConfig(SensorAttributesReference[OpenMeteoAttributes]): ...


class ForecastConfig(BaseModel):
    solcast: SolcastConfig = Field()
    open_meteo: OpenMeteoConfig = Field()


class Config(BaseModel):
    solar: SolarConfig = Field()
    heat_pump: HeatPumpConfig = Field()
    forecast: ForecastConfig = Field()


class FitConfig(BaseModel):
    forecaster: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class PredictConfig(BaseModel):
    forecaster: ForecasterType
    steps: int = Field(default=24)


class BacktestConfig(BaseModel):
    forecaster: ForecasterType
    days: int = Field(default=90)
    steps: int = Field(default=24)


class TuneConfig(BacktestConfig):
    trails: int = Field(default=10)
