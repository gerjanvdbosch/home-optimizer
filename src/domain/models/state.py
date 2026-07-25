from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FloatPoint(BaseModel):
    time: datetime
    value: float


class StringPoint(BaseModel):
    time: datetime
    value: str


class SolarForecastState(BaseModel):
    p10: list[FloatPoint] = Field(default_factory=list)
    p50: list[FloatPoint] = Field(default_factory=list)
    p90: list[FloatPoint] = Field(default_factory=list)

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class SolarState(BaseModel):
    production: list[FloatPoint] = Field(default_factory=list)
    forecast: SolarForecastState = Field(default_factory=SolarForecastState)


class BoilerState(BaseModel):
    top_temperature: list[FloatPoint] = Field(default_factory=list)
    bottom_temperature: list[FloatPoint] = Field(default_factory=list)


class HeatPumpState(BaseModel):
    mode: list[StringPoint] = Field(default_factory=list)
    supply_temperature: list[FloatPoint] = Field(default_factory=list)
    return_temperature: list[FloatPoint] = Field(default_factory=list)
    boiler: BoilerState = Field(default_factory=BoilerState)


class OptimizerState(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    solar: SolarState = Field(default_factory=SolarState)
    heat_pump: HeatPumpState = Field(default_factory=HeatPumpState)
