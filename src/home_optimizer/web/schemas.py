from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HistoryImportRunResponse(BaseModel):
    job_id: str
    status: str = Field(default="pending")
    sensor_count: int


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


class DashboardViewModel(BaseModel):
    title: str
    import_window_days: int
    chunk_days: int
    sensor_count: int
    database_path: str
    api_port: int


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
    dhw_temperatures: list[ChartSeriesResponse]
    heatpump_power: ChartSeriesResponse
    heatpump_mode: ChartTextSeriesResponse
    heatpump_statuses: list[ChartSeriesResponse]
    forecast_temperature: ChartSeriesResponse
    forecast_gti: list[ChartSeriesResponse]
