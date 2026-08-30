import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Iterable, Literal, Protocol, Sequence, TypeVar

import pandas as pd
from optuna import Study
from pandas._typing import MergeHow
from pydantic import BaseModel, Field, model_validator

HeatPumpMode = Literal[
    "heat",
    "cool",
]

ForecasterType = Literal[
    "solar",
    "boiler",
    "baseload",
]

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


class JobType(str, Enum):
    UPDATE = "update"
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"
    OPTIMIZE = "optimize"


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


JsonType = dict[str, Any] | list[Any]


class AttributeDefinition(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


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


T = TypeVar("T")


class SensorAttributesReference(BaseModel, Generic[T]):
    entity_id: str = Field()
    attributes: T

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attributes": {},
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attributes": value[1],
            }

        return value


class BoilerConfig(BaseModel):
    setpoint: SensorReference = Field()
    top_temperature: SensorReference = Field()
    bottom_temperature: SensorReference = Field()
    target_temperature: float | list[tuple[time, float]] = Field()


class HeatPumpConfig(BaseModel):
    state: SensorReference = Field()
    power: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    compressor_frequency: SensorReference = Field()
    boiler: BoilerConfig = Field()


class ClimateConfig(BaseModel):
    temperature: SensorReference = Field()
    setpoint: SensorReference = Field()
    target_temperature: float | list[tuple[time, float]] = Field()


class SolcastAttributes(BaseModel):
    p10: str = Field(default="pv_estimate10", description="10e percentile")
    p50: str = Field(default="pv_estimate", description="50e percentile")
    p90: str = Field(default="pv_estimate90", description="90e percentile")

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class SolcastConfig(SensorAttributesReference[SolcastAttributes]): ...


class OpenMeteoAttributes(BaseModel):
    temperature: str = Field(default="temperature_2m")
    is_day: str = Field(default="is_day")
    gti: str = Field(default="global_tilted_irradiance")
    cloud_cover_low: str = Field(default="cloud_cover_low")
    cloud_cover_mid: str = Field(default="cloud_cover_mid")
    cloud_cover_high: str = Field(default="cloud_cover_high")
    wind_direction: str = Field(default="wind_direction_10m")
    wind_speed: str = Field(default="wind_speed_10m")
    precipitation: str = Field(default="precipitation")

    def items(self):
        return (
            ("temperature", self.temperature),
            ("is_day", self.is_day),
            ("gti", self.gti),
            ("cloud_cover_low", self.cloud_cover_low),
            ("cloud_cover_mid", self.cloud_cover_mid),
            ("cloud_cover_high", self.cloud_cover_high),
            ("wind_direction", self.wind_direction),
            ("wind_speed", self.wind_speed),
            ("precipitation", self.precipitation),
        )


class OpenMeteoConfig(SensorAttributesReference[OpenMeteoAttributes]): ...


class ForecastConfig(BaseModel):
    solcast: SolcastConfig = Field()
    open_meteo: OpenMeteoConfig = Field()


class Config(BaseModel):
    solar: SensorReference = Field()
    baseload: SensorReference = Field()
    heat_pump: HeatPumpConfig = Field()
    climate: ClimateConfig = Field()
    forecast: ForecastConfig = Field()
    presence: list[SensorReference] = Field(default_factory=list)


class FitConfig(BaseModel):
    forecaster: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class PredictConfig(BaseModel):
    forecaster: ForecasterType | None = Field(default=None)
    steps: int = Field(default=48)


class BacktestConfig(BaseModel):
    forecaster: ForecasterType
    days: int = Field(default=90)
    steps: int = Field(default=24)


class TuneConfig(BacktestConfig):
    trails: int = Field(default=10)


class OptimizeConfig(BaseModel): ...


class MPCConfig(BaseModel):
    step_hours: float = Field(default=0.25, gt=0)
    boiler_power: float = Field(default=2.0, gt=0)
    boiler_duration_hours: float = Field(default=1.0, gt=0)
    max_starts: int = 1

    @property
    def boiler_steps(self) -> int:
        return int(self.boiler_duration_hours / self.step_hours)


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_kw: Sequence[float]
    boiler_on: bool = False


@dataclass(frozen=True)
class MPCResult:
    schedule: tuple[int, ...]
    objective_value: float
    solver_status: str
    termination_condition: str


class DataDefinition(BaseModel):
    name: str


class TimeSeriesDefinition(DataDefinition):
    sensor: SensorReference
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod | int | float = "none"


class AttributeSeriesDefinition(DataDefinition):
    sensor: SensorAttributesReference
    attributes: list[str]
    time_attribute: str = "time"
    target_interval: str | None = None
    target_closed: Literal["right", "left"] | None = None
    target_label: Literal["right", "left"] | None = None
    target_resample: str | None = None
    target_shift: bool | list[str] = False


class AttributeTimeSeriesDefinition(AttributeSeriesDefinition):
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod | int | float = "none"


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


P = TypeVar("P")


class SeriesPoint(BaseModel, Generic[P]):
    time: datetime
    value: P


class BoilerMeasurement(BaseModel):
    top_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    bottom_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpMeasurement(BaseModel):
    mode: HeatPumpMode = "heat"
    state: list[SeriesPoint[str]] = Field(default_factory=list)
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    supply_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    return_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    compressor_frequency: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerMeasurement = Field(default_factory=BoilerMeasurement)


class ClimateMeasurement(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    setpoint: list[SeriesPoint[float]] = Field(default_factory=list)


class Measurements(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    heat_pump: HeatPumpMeasurement = Field(default_factory=HeatPumpMeasurement)
    climate: ClimateMeasurement = Field(default_factory=ClimateMeasurement)


class ElectricityPriceForecast(BaseModel):
    price: list[SeriesPoint[float]] = Field(default_factory=list)


class SolcastForecast(BaseModel):
    p10: list[SeriesPoint[float]] = Field(default_factory=list)
    p50: list[SeriesPoint[float]] = Field(default_factory=list)
    p90: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class OpenMeteoForecast(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    gti: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_low: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_mid: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_high: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_direction: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_speed: list[SeriesPoint[float]] = Field(default_factory=list)
    precipitation: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("temperature", self.temperature),
            ("gti", self.gti),
            ("cloud_cover_low", self.cloud_cover_low),
            ("cloud_cover_mid", self.cloud_cover_mid),
            ("cloud_cover_high", self.cloud_cover_high),
            ("wind_direction", self.wind_direction),
            ("wind_speed", self.wind_speed),
            ("precipitation", self.precipitation),
        )


class Forecast(BaseModel):
    solcast: SolcastForecast = Field(default_factory=SolcastForecast)
    open_meteo: OpenMeteoForecast = Field(default_factory=OpenMeteoForecast)
    electricity_price: ElectricityPriceForecast = Field(
        default_factory=ElectricityPriceForecast
    )


class Predictions(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpSchedule(BaseModel):
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerSchedule = Field(default_factory=BoilerSchedule)


class ClimateSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class Schedule(BaseModel):
    heat_pump: HeatPumpSchedule = Field(default_factory=HeatPumpSchedule)
    climate: ClimateSchedule = Field(default_factory=ClimateSchedule)


class State(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    measurements: Measurements = Field(default_factory=Measurements)
    forecast: Forecast = Field(default_factory=Forecast)
    predictions: Predictions = Field(default_factory=Predictions)
    schedule: Schedule = Field(default_factory=Schedule)


class BacktestPoint(BaseModel):
    label: str
    points: list[dict[str, object]]
    color: str | None = None
    group: str | None = None


class BacktestResult(BaseModel):
    name: ForecasterType
    label: str
    unit: str
    mae: float
    points: list[BacktestPoint]


@dataclass
class Job:
    type: JobType
    config: (
        Config
        | FitConfig
        | PredictConfig
        | TuneConfig
        | BacktestConfig
        | OptimizeConfig
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


D = TypeVar("D", bound=DataDefinition, contravariant=True)


class DataLoader(Protocol[D]):
    def supports(self, definition: DataDefinition) -> bool: ...

    def load(self, definition: D, start: datetime, end: datetime) -> pd.DataFrame: ...


class Forecaster(Protocol):
    @property
    def name(self) -> ForecasterType: ...

    @property
    def target_column(self) -> str: ...

    @property
    def exog_columns(self) -> list[str]: ...

    @property
    def label(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def dataset(self, config: Config) -> DatasetDefinition: ...

    def fit(self, df: pd.DataFrame): ...

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.Series: ...

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult: ...

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path, study_storage: str) -> None: ...
