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


class WeatherSegmentResponse(BaseModel):
    start: str
    end: str
    code: int
    label: str


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
    forecast_precipitation: ChartSeriesResponse
    forecast_weather_segments: list[WeatherSegmentResponse]
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
    space_heating_output_estimate_kw: float | None = None
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


class HorizonMetricResponse(BaseModel):
    horizon_steps: int
    horizon_minutes: int
    sample_count: int
    mae_c: float | None = None
    rmse_c: float | None = None
    bias_c: float | None = None
    p95_abs_error_c: float | None = None


class SegmentValidationResponse(BaseModel):
    segment_name: str
    description: str
    metrics: list[HorizonMetricResponse]


class TrainRoomModelResponse(BaseModel):
    model_id: str
    model_type: str
    created_at_utc: datetime
    trained_from_utc: datetime
    trained_to_utc: datetime
    validation_from_utc: datetime | None = None
    validation_to_utc: datetime | None = None
    test_from_utc: datetime | None = None
    test_to_utc: datetime | None = None
    interval_minutes: int
    sample_count: int
    is_active: bool
    fit_quality: str | None = None
    fit_quality_reasons: list[str] = []
    aggregate_metrics: list[HorizonMetricResponse]
    segment_metrics: list[SegmentValidationResponse]
    test_aggregate_metrics: list[HorizonMetricResponse] = []
    test_segment_metrics: list[SegmentValidationResponse] = []


class RoomModelVersionSummaryResponse(BaseModel):
    model_id: str
    model_type: str
    created_at_utc: datetime
    trained_from_utc: datetime
    trained_to_utc: datetime
    interval_minutes: int
    sample_count: int
    is_active: bool
    validation_mae_1h_c: float | None = None
    validation_mae_6h_c: float | None = None
    validation_mae_12h_c: float | None = None
    validation_mae_24h_c: float | None = None
    validation_bias_6h_c: float | None = None
    validation_p95_12h_c: float | None = None


class RoomModelCatalogResponse(BaseModel):
    models: list[RoomModelVersionSummaryResponse]


class RoomModelVersionDetailResponse(BaseModel):
    model_id: str
    model_type: str
    created_at_utc: datetime
    trained_from_utc: datetime
    trained_to_utc: datetime
    interval_minutes: int
    sample_count: int
    is_active: bool
    fit_quality: str | None = None
    fit_quality_reasons: list[str] = []
    aggregate_metrics: list[HorizonMetricResponse] = []
    segment_metrics: list[SegmentValidationResponse] = []


class RoomSimulationResponse(BaseModel):
    model_id: str
    anchor_time_utc: datetime
    interval_minutes: int
    horizon_steps: int
    predicted_room_temperature: ChartSeriesResponse
    actual_room_temperature: ChartSeriesResponse
    prediction_error_c: ChartSeriesResponse
    room_target_min_temperature: ChartSeriesResponse
    room_target_max_temperature: ChartSeriesResponse
    outdoor_temperature: ChartSeriesResponse
    thermal_output_estimate: ChartSeriesResponse
    solar_irradiance: ChartSeriesResponse
    solar_gain_proxy: ChartSeriesResponse
    shutter_position: ChartSeriesResponse


class MpcPlanStepResponse(BaseModel):
    timestamp_utc: datetime
    hp_on: bool
    start: bool
    stop: bool
    predicted_room_temp_c: float
    economic_target_c: float
    useful_preheat_target_c: float
    preheat_active: bool
    preheat_block_id: int | None = None
    preheat_opportunity_score: float
    preheat_budget_share_kwh: float
    preheat_charge_kwh: float
    preheat_block_budget_kwh: float
    q_heat_eff_kw: float
    temp_min_c: float
    temp_max_c: float
    slack_low_c: float
    slack_high_c: float
    effective_heating_kw: float
    price_eur_kwh: float
    estimated_energy_cost_eur: float


class MpcObjectiveBreakdownResponse(BaseModel):
    comfort_low: float
    active_comfort_high_cost: float
    passive_comfort_high_cost: float
    comfort_high: float
    comfort_total: float
    tracking_under_target: float
    tracking_over_target: float
    temperature_tracking: float
    energy_cost: float
    pv_self_consumption_reward: float
    captured_pv_kwh: float
    preheat_budget_shortfall: float
    unnecessary_heating: float
    terminal_cost: float
    start_penalty: float
    runtime: float
    total: float


class MpcPlanSummaryResponse(BaseModel):
    step_count: int
    start_count: int
    stop_count: int
    comfort_violation_count: int
    slack_usage_count: int
    runtime_steps: int
    estimated_energy_cost_eur: float


class MpcPlanResponse(BaseModel):
    control_mode: str = "hierarchical_preheat"
    status: str
    termination_condition: str
    feasible: bool
    objective_value: float | None = None
    solve_time_seconds: float | None = None
    heating_explanation: str | None = None
    objective_breakdown: MpcObjectiveBreakdownResponse
    summary: MpcPlanSummaryResponse
    steps: list[MpcPlanStepResponse]


class MpcBacktestSummaryResponse(BaseModel):
    comfort_violation_minutes: int
    degree_minutes_below_comfort: float
    degree_minutes_above_comfort: float
    active_comfort_high_degree_minutes: float
    passive_comfort_high_degree_minutes: float
    starts_per_day: float
    runtime_minutes: int
    estimated_energy_cost_eur: float
    average_solver_runtime_seconds: float
    infeasible_count: int
    slack_usage_count: int


class MpcBacktestPvDiagnosticsResponse(BaseModel):
    realized_pv_surplus_kwh: float
    forecast_pv_surplus_kwh: float
    mpc_hp_energy_kwh: float
    mpc_hp_energy_during_realized_pv_surplus_kwh: float
    mpc_hp_energy_during_forecast_pv_surplus_kwh: float
    mpc_realized_pv_surplus_capture_kwh: float
    mpc_realized_pv_surplus_capture_ratio: float
    mpc_forecast_pv_surplus_capture_ratio: float
    preheat_budget_electric_kwh: float
    used_preheat_budget_kwh: float
    missed_surplus_with_headroom_kwh: float
    captured_realized_pv_kwh: float
    capture_ratio_realized: float
    average_run_duration_minutes: float
    short_run_count: int
    preheat_block_count: int
    starts_per_preheat_block: float


class MpcBacktestDeltaResponse(BaseModel):
    comfort_violation_minutes: int
    degree_minutes_below_comfort: float
    degree_minutes_above_comfort: float
    active_comfort_high_degree_minutes: float
    passive_comfort_high_degree_minutes: float
    starts_per_day: float
    runtime_minutes: int
    estimated_energy_cost_eur: float
    average_solver_runtime_seconds: float
    infeasible_count: int
    slack_usage_count: int


class MpcBacktestStepResponse(BaseModel):
    timestamp_utc: datetime
    forecast_issue_time_utc: datetime | None = None
    forecast_age_minutes: float
    mpc_hp_on: bool
    historical_hp_on: bool
    start: bool
    stop: bool
    preheat_active: bool
    preheat_block_id: int | None = None
    preheat_budget_share_kwh: float
    preheat_charge_kwh: float
    preheat_opportunity_score: float
    q_heat_eff_kw: float
    historical_q_heat_eff_kw: float
    hp_electric_power_kw: float
    pv_forecast_kw: float
    pv_realized_kw: float
    solar_irradiance_forecast_wm2: float
    solar_irradiance_realized_wm2: float
    solar_gain_forecast_kw: float
    solar_gain_realized_kw: float
    base_load_forecast_kw: float
    base_load_realized_kw: float
    pv_surplus_forecast_kw: float
    pv_surplus_realized_kw: float
    predicted_next_room_temp_c: float
    simulated_next_room_temp_c: float
    historical_next_room_temp_c: float | None = None
    temp_min_c: float
    temp_max_c: float
    slack_low_c: float
    slack_high_c: float
    price_eur_kwh: float
    estimated_mpc_energy_cost_eur: float
    estimated_historical_energy_cost_eur: float
    solve_time_seconds: float | None = None
    feasible: bool


class MpcBacktestResponse(BaseModel):
    exogenous_mode: str
    control_mode: str
    missing_forecast_count: int
    forecast_coverage_ratio: float
    model_id: str
    model_type: str
    start_time_utc: datetime
    end_time_utc: datetime
    interval_minutes: int
    horizon_steps: int
    step_count: int
    mpc_objective_breakdown: MpcObjectiveBreakdownResponse
    solver_objective_breakdown: MpcObjectiveBreakdownResponse
    mpc_summary: MpcBacktestSummaryResponse
    historical_summary: MpcBacktestSummaryResponse
    pv_diagnostics: MpcBacktestPvDiagnosticsResponse
    delta: MpcBacktestDeltaResponse
    total_solver_runtime_seconds: float
    steps: list[MpcBacktestStepResponse]
