from __future__ import annotations

from datetime import datetime

from pydantic import Field

from home_optimizer.domain.models import DomainModel
from home_optimizer.features.mpc.models import MpcObjectiveBreakdown
from home_optimizer.features.mpc_new.models import (
    AuthorityInvariantReport,
    StartReason,
    StartStopLedgerEntry,
    StopReason,
)


class MpcBacktestPvDiagnostics(DomainModel):
    realized_pv_surplus_kwh: float = 0.0
    forecast_pv_surplus_kwh: float = 0.0
    mpc_hp_energy_kwh: float = 0.0
    mpc_hp_energy_during_realized_pv_surplus_kwh: float = 0.0
    mpc_hp_energy_during_forecast_pv_surplus_kwh: float = 0.0
    mpc_realized_pv_surplus_capture_kwh: float = 0.0
    mpc_realized_pv_surplus_capture_ratio: float = 0.0
    mpc_forecast_pv_surplus_capture_ratio: float = 0.0
    preheat_budget_electric_kwh: float = 0.0
    used_preheat_budget_kwh: float = 0.0
    missed_surplus_with_headroom_kwh: float = 0.0
    captured_realized_pv_kwh: float = 0.0
    capture_ratio_realized: float = 0.0
    average_run_duration_minutes: float = 0.0
    short_run_count: int = 0
    preheat_block_count: int = 0
    starts_per_preheat_block: float = 0.0


class MpcBacktestStepResult(DomainModel):
    timestamp_utc: datetime
    forecast_issue_time_utc: datetime | None = None
    forecast_age_minutes: float = 0.0
    mpc_hp_on: bool
    historical_hp_on: bool
    start: bool
    stop: bool
    start_reason: StartReason | None = None
    stop_reason: StopReason | None = None
    planned_room_temp_c: float = 0.0
    useful_preheat_target_c: float = 0.0
    preheat_active: bool = False
    preheat_block_id: int | None = None
    preheat_budget_share_kwh: float = 0.0
    preheat_charge_kwh: float = 0.0
    preheat_opportunity_score: float = 0.0
    q_heat_eff_kw: float = 0.0
    historical_q_heat_eff_kw: float = 0.0
    hp_electric_power_kw: float = 0.0
    pv_forecast_kw: float = 0.0
    pv_realized_kw: float = 0.0
    solar_irradiance_forecast_wm2: float = 0.0
    solar_irradiance_realized_wm2: float = 0.0
    solar_gain_forecast_kw: float = 0.0
    solar_gain_realized_kw: float = 0.0
    base_load_forecast_kw: float = 0.0
    base_load_realized_kw: float = 0.0
    pv_surplus_forecast_kw: float = 0.0
    pv_surplus_realized_kw: float = 0.0
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
    feasible: bool = True


class MpcBacktestSummary(DomainModel):
    comfort_violation_minutes: int
    degree_minutes_below_comfort: float
    degree_minutes_above_comfort: float
    active_comfort_high_degree_minutes: float = 0.0
    passive_comfort_high_degree_minutes: float = 0.0
    starts_per_day: float
    runtime_minutes: int
    estimated_energy_cost_eur: float
    average_solver_runtime_seconds: float = 0.0
    infeasible_count: int = 0
    slack_usage_count: int = 0


class MpcBacktestResult(DomainModel):
    exogenous_mode: str = "perfect_foresight"
    control_mode: str = "hierarchical_preheat"
    missing_forecast_count: int = 0
    forecast_coverage_ratio: float = 1.0
    model_id: str
    model_type: str
    start_time_utc: datetime
    end_time_utc: datetime
    interval_minutes: int
    horizon_steps: int
    step_results: list[MpcBacktestStepResult]
    mpc_summary: MpcBacktestSummary
    historical_summary: MpcBacktestSummary
    pv_diagnostics: MpcBacktestPvDiagnostics = Field(
        default_factory=MpcBacktestPvDiagnostics
    )
    start_stop_ledger: list[StartStopLedgerEntry] = Field(default_factory=list)
    invariant_report: AuthorityInvariantReport = Field(
        default_factory=AuthorityInvariantReport
    )
    mpc_objective_breakdown: MpcObjectiveBreakdown
    solver_objective_breakdown: MpcObjectiveBreakdown
    total_solver_runtime_seconds: float
