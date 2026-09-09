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
from scipy.linalg import expm

HeatPumpMode = Literal["heat", "cool"]

ForecasterType = Literal["solar", "baseload"]

IdentificationType = Literal["boiler"]


class JobType(str, Enum):
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
    gti: str = Field(default="global_tilted_irradiance")
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
            ("gti", self.gti),
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
    temperatures_top: list[SeriesPoint[float]] = Field(default_factory=list)
    temperatures_bottom: list[SeriesPoint[float]] = Field(default_factory=list)


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
    """
    Reduced-order thermal model of a stratified 200 L DHW boiler.

    The model has three thermal states:
        T_top     : top-layer temperature [°C]
        T_mid     : lumped middle-layer temperature [°C]
        T_bottom  : bottom-layer temperature [°C]

    Only T_top and T_bottom are assumed to be measured directly.
    T_mid is an estimated/hidden state.
    """

    # Thermal conductance between the three nodes [W/K]
    K12: float
    K23: float

    # Heat-loss conductance of each node to ambient [W/K]
    G1: float
    G2: float
    G3: float

    # Effective fraction of heater power transferred to the water [-]
    eta: float

    # Initial/estimated middle-node temperature [°C]
    T_mid_0: float

    # Identification performance
    rmse_top: float
    rmse_bottom: float

    # Independent validation performance
    validation_rmse_top: float
    validation_rmse_bottom: float

    @property
    def volume_l(self) -> float:
        return 200.0

    @property
    def rho(self) -> float:
        return 997.0

    @property
    def cp(self) -> float:
        return 4180.0

    @property
    def total_capacity_j_per_k(self) -> float:
        return self.volume_l / 1000.0 * self.rho * self.cp

    @property
    def node_capacity_j_per_k(self) -> float:
        return self.total_capacity_j_per_k / 3.0

    def discrete_matrices(
        self,
        dt_seconds: float,
        flow_lpm: float,
    ):
        """
        Discretiseert het 3-state thermische model met exact ZOH.

        States:
            x = [T_top, T_mid, T_bottom]

        Inputs:
            u_heater   = boilervermogen [kW]
            T_cold     = koudwater [°C]
            T_ambient  = omgeving [°C]

        Returns:
            A, B, E_cold, E_ambient
        """
        C = self.node_capacity_j_per_k

        # kg/s
        mass_flow = (flow_lpm / 1000.0) * self.rho / 60.0

        mc = mass_flow * self.cp

        A_c = np.array(
            [
                [
                    -(self.K12 + self.G1 + mc) / C,
                    (self.K12 + mc) / C,
                    0.0,
                ],
                [
                    self.K12 / C,
                    -(self.K12 + self.K23 + self.G2 + mc) / C,
                    (self.K23 + mc) / C,
                ],
                [
                    0.0,
                    self.K23 / C,
                    -(self.K23 + self.G3 + mc) / C,
                ],
            ]
        )

        # heater input is expressed in kW
        B_c = np.array(
            [
                [0.0],
                [0.0],
                [self.eta * 1000.0 / C],
            ]
        )

        E_cold_c = np.array(
            [
                [0.0],
                [0.0],
                [mc / C],
            ]
        )

        E_ambient_c = np.array(
            [
                [self.G1 / C],
                [self.G2 / C],
                [self.G3 / C],
            ]
        )

        # Exact zero-order-hold discretisation.
        B_c_combined = np.hstack(
            [
                B_c,
                E_cold_c,
                E_ambient_c,
            ]
        )

        augmented = np.zeros((6, 6))

        augmented[:3, :3] = A_c
        augmented[:3, 3:] = B_c_combined

        exp_augmented = expm(augmented * dt_seconds)

        A = exp_augmented[:3, :3]
        B_combined = exp_augmented[:3, 3:]

        B = B_combined[:, [0]]
        E_cold = B_combined[:, [1]]
        E_ambient = B_combined[:, [2]]

        return A, B, E_cold, E_ambient


@dataclass
class HeatPumpCOPModel:
    eta_carnot: float
    delta_t_cond: float
    delta_t_evap: float


@dataclass(frozen=True)
class MPCConfig:
    """
    Configuration for DHW MPC.

    The objective prioritizes:
    1. satisfying top-temperature demand,
    2. maintaining a minimum bottom temperature,
    3. using available solar energy,
    4. avoiding unnecessary boiler starts,
    5. minimizing boiler energy.

    Temperature requirements are modeled as soft lower-bound constraints.
    """

    # MPC timestep
    step_hours: float = 5.0 / 60.0

    # Boiler / heat-pump thermal power
    boiler_power_w: float = 3000

    # Minimum number of consecutive MPC steps when boiler starts
    boiler_min_runtime_steps: int = 3

    # Absolute safety floor for the top of the tank
    boiler_min_top_temperature: float = 45.0

    # Minimum useful reserve temperature at the bottom
    boiler_min_bottom_temperature: float = 35.0

    # Objective weights
    weight_energy: float = 0.05
    weight_solar_priority: float = 1.0
    weight_switching: float = 0.5
    weight_temperature_slack: float = 1000.0


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_w: list[float]
    ambient_temperature: float
    current_temp_top: float
    current_temp_bottom: float
    boiler_on: bool
    thermal_model: BoilerThermalModel
    target_temperature_top: tuple[float, ...] = ()


@dataclass(frozen=True)
class MPCResult(BaseModel):
    schedule: tuple[int, ...]
    temperatures_top: tuple[float, ...]
    temperatures_bottom: tuple[float, ...]
    objective_value: float
    solver_status: str
    termination_condition: str
