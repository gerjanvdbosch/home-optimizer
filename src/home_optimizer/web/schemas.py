from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


class HistoryImportRunResponse(BaseModel):
    job_id: str
    status: str = Field(default="pending")
    sensor_count: int


class WeatherImportResponse(BaseModel):
    imported_rows: int


class HistoryImportJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    imported_rows: dict[str, int]
    total_rows: int
    sensor_count: int
    error: str | None


@dataclass(slots=True)
class DashboardPageViewModel:
    title: str
    database_path: str


class ChartPointResponse(BaseModel):
    timestamp: str
    value: float


class ChartSeriesResponse(BaseModel):
    name: str
    unit: str | None
    points: list[ChartPointResponse]


class ChartTextPointResponse(BaseModel):
    timestamp: str
    value: str


class ChartTextSeriesResponse(BaseModel):
    name: str
    points: list[ChartTextPointResponse]


class DashboardChartsResponse(BaseModel):
    date: str
    room_temperature: ChartSeriesResponse
    thermostat_setpoint: ChartSeriesResponse
    room_target_temperature: ChartSeriesResponse
    room_target_min_temperature: ChartSeriesResponse
    room_target_max_temperature: ChartSeriesResponse
    shutter_position: ChartSeriesResponse
    dhw_temperatures: list[ChartSeriesResponse]
    dhw_target_temperature: ChartSeriesResponse
    dhw_target_min_temperature: ChartSeriesResponse
    dhw_target_max_temperature: ChartSeriesResponse
    heatpump_power: ChartSeriesResponse
    heatpump_mode: ChartTextSeriesResponse
    heatpump_statuses: list[ChartSeriesResponse]
    forecast_temperature: ChartSeriesResponse
    forecast_gti: list[ChartSeriesResponse]
    historical_weather_temperature: ChartSeriesResponse
    historical_weather_gti: list[ChartSeriesResponse]
    pv_output_power: ChartSeriesResponse
    baseload: ChartSeriesResponse
    thermal_output: ChartSeriesResponse
    cop: ChartSeriesResponse
    hp_supply_temperature: ChartSeriesResponse
    hp_supply_target_temperature: ChartSeriesResponse
    hp_return_temperature: ChartSeriesResponse
    hp_delta_t: ChartSeriesResponse
    hp_flow: ChartSeriesResponse
    compressor_frequency: ChartSeriesResponse

