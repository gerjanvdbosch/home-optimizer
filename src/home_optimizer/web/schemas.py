from __future__ import annotations

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
    thermostat_setpoint: ChartSeriesResponse
    shutter_position: ChartSeriesResponse
    dhw_temperatures: list[ChartSeriesResponse]
    heatpump_power: ChartSeriesResponse
    heatpump_mode: ChartTextSeriesResponse
    heatpump_statuses: list[ChartSeriesResponse]
    forecast_temperature: ChartSeriesResponse
    forecast_gti: list[ChartSeriesResponse]
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


class IdentificationResponse(BaseModel):
    model_name: str
    interval_minutes: int
    sample_count: int
    train_sample_count: int
    test_sample_count: int
    coefficients: dict[str, float]
    intercept: float
    train_rmse: float
    test_rmse: float
    test_rmse_recursive: float
    target_name: str


class IdentificationTrainRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    interval_minutes: int = Field(default=15, ge=1)
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)


class StoredIdentifiedModelResponse(BaseModel):
    model_name: str
    trained_at_utc: datetime
    training_start_time_utc: datetime
    training_end_time_utc: datetime
    interval_minutes: int
    sample_count: int
    train_sample_count: int
    test_sample_count: int
    coefficients: dict[str, float]
    intercept: float
    train_rmse: float
    test_rmse: float
    test_rmse_recursive: float
    target_name: str


class NumericSeriesRequestPoint(BaseModel):
    timestamp: str
    value: float


class NumericSeriesRequest(BaseModel):
    name: str
    unit: str | None
    points: list[NumericSeriesRequestPoint]


class PredictionRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    thermostat_schedule: NumericSeriesRequest
    shutter_schedule: NumericSeriesRequest | None = None


class PredictionResponse(BaseModel):
    model_name: str
    interval_minutes: int
    target_name: str
    room_temperature: ChartSeriesResponse


class PredictionComparisonResponse(BaseModel):
    model_name: str
    interval_minutes: int
    target_name: str
    predicted_room_temperature: ChartSeriesResponse
    actual_room_temperature: ChartSeriesResponse
    overlap_count: int
    rmse: float | None
    bias: float | None
    max_absolute_error: float | None
