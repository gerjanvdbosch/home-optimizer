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
    electricity_price: ChartSeriesResponse
    room_temperature: ChartSeriesResponse
    outdoor_temperature: ChartSeriesResponse
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


class DailyKpiResponse(BaseModel):
    is_valid_for_control_evaluation: bool
    validity_reasons: list[str]
    data_coverage_pct: float | None
    largest_data_gap_minutes: float | None
    hp_electric_kwh: float | None
    total_import_kwh: float | None
    total_export_kwh: float | None
    pv_generation_kwh: float | None
    solar_irradiance_mean_w_m2: float | None
    shutter_open_pct_mean: float | None
    outdoor_temperature_mean_c: float | None
    self_consumption_ratio: float | None
    electricity_cost_eur: float | None
    room_temperature_mae_c: float | None
    room_comfort_undershoot_degree_hours: float | None
    comfort_overshoot_while_heating_degree_hours: float | None
    comfort_overshoot_passive_degree_hours: float | None
    dhw_comfort_undershoot_minutes: float | None
    thermostat_setpoint_changes: int
    compressor_starts: int


class BaselineKpiSummaryResponse(BaseModel):
    number_of_days: int
    number_of_valid_days: int
    mean_hp_electric_kwh_per_day: float | None
    mean_electricity_cost_eur_per_day: float | None
    mean_room_temperature_mae_c: float | None
    mean_solar_irradiance_w_m2: float | None
    mean_shutter_open_pct: float | None
    total_comfort_undershoot_degree_hours: float
    total_comfort_overshoot_while_heating_degree_hours: float
    total_comfort_overshoot_passive_degree_hours: float
    total_dhw_undershoot_minutes: float
    mean_compressor_starts_per_day: float | None
    mean_self_consumption_ratio: float | None


class IdentificationDatasetRowResponse(BaseModel):
    timestamp_utc: datetime
    room_temperature_c: float | None = None
    outdoor_temperature_c: float | None = None
    dhw_top_temperature_c: float | None = None
    dhw_bottom_temperature_c: float | None = None
    hp_electric_power_kw: float | None = None
    hp_mode_raw: str | None = None
    mode_space: int
    mode_dhw: int
    mode_off: int
    defrost_active: int
    booster_heater_active: int
    pv_output_power_kw: float | None = None
    net_power_kw: float | None = None
    shutter_position_pct: float | None = None
    thermostat_setpoint_c: float | None = None
    room_target_temperature_c: float | None = None
    room_target_min_temperature_c: float | None = None
    room_target_max_temperature_c: float | None = None
    supply_temperature_c: float | None = None
    return_temperature_c: float | None = None
    flow_l_min: float | None = None
    hp_delta_t_c: float | None = None
    thermal_output_estimate_kw: float | None = None
    cop_estimate: float | None = None
    solar_irradiance_w_m2: float | None = None
    solar_gain_proxy_w_m2: float | None = None
    price_import_eur_kwh: float | None = None
    price_export_eur_kwh: float | None = None
    occupied_flag: int
    dhw_draw_proxy_c: float
    dhw_draw_detected: int
    is_valid_for_room_identification: bool
    is_valid_for_dhw_identification: bool
    is_valid_for_cop_identification: bool
    exclusion_reasons: list[str]


class IdentificationDatasetSummaryResponse(BaseModel):
    total_rows: int
    mode_space_rows: int
    mode_dhw_rows: int
    mode_off_rows: int
    defrost_rows: int
    booster_rows: int
    valid_room_rows: int
    valid_dhw_rows: int
    valid_cop_rows: int
    exclusion_reason_counts: dict[str, int]


class IdentificationDatasetResponse(BaseModel):
    interval_minutes: int
    start_time_utc: datetime
    end_time_utc: datetime
    summary: IdentificationDatasetSummaryResponse
    rows: list[IdentificationDatasetRowResponse]
