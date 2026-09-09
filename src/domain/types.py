import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
import pandas as pd
from optuna import Study
from pydantic import BaseModel, Field, model_validator

HeatPumpMode = Literal["heat", "cool"]

ForecasterType = Literal["solar", "baseload", "tap"]

IdentificationType = Literal["boiler"]


class JobType(str, Enum):
    CONFIG = "config"
    UPDATE = "update"
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"
    CALIBRATE = "calibrate"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


JsonType = dict[str, Any] | list[Any]


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
    ambient_temperature: SensorReference = Field()
    volume: int = Field(default=200)
    target_temperature: float | list[tuple[time, float]] = Field()


class HeatPumpConfig(BaseModel):
    state: SensorReference = Field()
    power: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    compressor_frequency: SensorReference = Field()
    flow: SensorReference = Field()
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
    global_tilted_irradiance: str = Field(default="global_tilted_irradiance")
    direct_radiation: str = Field(default="direct_radiation")
    direct_normal_irradiance: str = Field(default="direct_normal_irradiance")
    diffuse_radiation: str = Field(default="diffuse_radiation")
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
            ("global_tilted_irradiance", self.global_tilted_irradiance),
            ("direct_radiation", self.direct_radiation),
            ("direct_normal_irradiance", self.direct_normal_irradiance),
            ("diffuse_radiation", self.diffuse_radiation),
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


class UpdateConfig(BaseModel): ...


class FitConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class PredictConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    steps: int = Field(default=48)


class BacktestConfig(BaseModel):
    target: ForecasterType
    days: int = Field(default=90)
    steps: int = Field(default=24)


class TuneConfig(BacktestConfig):
    trails: int = Field(default=10)


class CalibrateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)


class ValidateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)
    steps: int = Field(default=8)


class OptimizeConfig(BaseModel): ...


P = TypeVar("P")


class SeriesPoint(BaseModel, Generic[P]):
    time: datetime
    value: P


class BoilerMeasurement(BaseModel):
    top_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    bottom_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    ambient_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


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
    global_tilted_irradiance: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_low: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_mid: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_high: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_direction: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_speed: list[SeriesPoint[float]] = Field(default_factory=list)
    precipitation: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("temperature", self.temperature),
            ("global_tilted_irradiance", self.global_tilted_irradiance),
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


class Predictions(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    tap: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    temperatures: list[SeriesPoint[float]] = Field(default_factory=list)


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
    rmse: float
    r2: float
    points: list[BacktestPoint]


@dataclass
class Job:
    type: JobType
    config: (
        Config
        | UpdateConfig
        | FitConfig
        | PredictConfig
        | TuneConfig
        | BacktestConfig
        | CalibrateConfig
        | ValidateConfig
        | OptimizeConfig
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class BoilerThermalModel:
    volume_l: float
    ua_top_w_per_k: float
    ua_bottom_w_per_k: float
    ua_mix_idle_w_per_k: float
    ua_mix_active_w_per_k: float
    q_in_nominal_w: float


@dataclass
class HeatPumpCOPModel:
    eta_carnot: float
    delta_t_cond: float
    delta_t_evap: float


@dataclass(frozen=True)
class MPCConfig:
    step_hours: float = 0.25
    # No calibrated COP model is wired in yet (HeatPumpCOPModel exists, unused) - an
    # explicit, stated electrical-power assumption for costing, kept separate from
    # the calibrated thermal q_in_nominal_w used for the temperature dynamics.
    boiler_electrical_power_w: float = 3000.0
    boiler_min_runtime_steps: int = 2
    # Flat price for now - will become a per-installation config option later.
    price_eur_per_kwh: float = 0.23
    weight_switching: float = 0.5
    weight_temperature_slack: float = 1000.0


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_w: list[float]
    # Held constant across the horizon: the boiler's local ambient sensor has no
    # forecast (unlike outdoor temperature, which has Open-Meteo) and is indoors,
    # where conditions change slowly relative to a typical MPC horizon.
    ambient_temperature: float
    current_temp_top: float
    current_temp_bottom: float
    boiler_on_current: bool
    thermal_model: BoilerThermalModel
    target_temperature_top: tuple[float, ...] = ()
    # Forecasted additional heat-sink power (W) from tap draws (see
    # features/tap.py's TapForecaster), on top of the passive UA loss already in
    # the dynamics - empty means "no forecast available", treated as no draws
    # (the same assumption implicitly made before this field existed), not a
    # claim that none will occur. This forecast has known, real but modest
    # accuracy (see TapForecaster's own backtest) - weight_temperature_slack
    # absorbs the resulting forecast error, same as it already does for solar.
    # It also targets a quantity that is a mix of real tap draws and a known,
    # uncorrected temperature-dependent heat-transfer gap in the passive-loss
    # model (see BoilerThermalIdentifier.excess_loss_w's docstring) - not tap
    # draws alone.
    tap_forecast_w: tuple[float, ...] = ()


@dataclass(frozen=True)
class MPCResult:
    schedule: tuple[int, ...]
    temperatures: tuple[float, ...]
    objective_value: float
    solver_status: str
    termination_condition: str
