from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from domain.models.config import HeatPumpMode

T = TypeVar("T")


class SeriesPoint(BaseModel, Generic[T]):
    time: datetime
    value: T


class SolarMeasurements(BaseModel):
    production: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerMeasurements(BaseModel):
    top_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    bottom_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpMeasurements(BaseModel):
    mode: HeatPumpMode = "heat"
    state: list[SeriesPoint[str]] = Field(default_factory=list)
    supply_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    return_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    compressor_frequency: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerMeasurements = Field(default_factory=BoilerMeasurements)


class MeasurementState(BaseModel):
    solar: SolarMeasurements = Field(default_factory=SolarMeasurements)
    heat_pump: HeatPumpMeasurements = Field(default_factory=HeatPumpMeasurements)


class SolarForecast(BaseModel):
    p10: list[SeriesPoint[float]] = Field(default_factory=list)
    p50: list[SeriesPoint[float]] = Field(default_factory=list)
    p90: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class ElectricityPriceForecast(BaseModel):
    price: list[SeriesPoint[float]] = Field(default_factory=list)


class WeatherForecast(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class ForecastState(BaseModel):
    solar: SolarForecast = Field(default_factory=SolarForecast)
    electricity_price: ElectricityPriceForecast = Field(default_factory=ElectricityPriceForecast)
    weather: WeatherForecast = Field(default_factory=WeatherForecast)


class HeatPumpSchedule(BaseModel):
    power: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class ScheduleState(BaseModel):
    boiler: BoilerSchedule = Field(default_factory=BoilerSchedule)


class OptimizerState(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    measurements: MeasurementState = Field(default_factory=MeasurementState)
    forecast: ForecastState = Field(default_factory=ForecastState)
    schedule: ScheduleState | None = None
