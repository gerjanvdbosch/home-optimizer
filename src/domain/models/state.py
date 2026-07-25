from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FloatPoint(BaseModel):
    time: datetime
    value: float


class BoilerState(BaseModel):
    temperature_top: list[FloatPoint]
    temperature_bottom: list[FloatPoint]


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


class OptimizerState(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    solar_forecast: SolarForecastState = Field(default_factory=SolarForecastState)
    pv_production: list[FloatPoint] = Field(default_factory=list)
    boiler: list[BoilerState] = Field(default_factory=list)
