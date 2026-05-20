from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from home_optimizer.domain.models import DomainModel
from home_optimizer.features.mpc.models import (
    MpcConstraints,
    MpcHorizonStep,
    MpcInitialState,
    MpcObjectiveBreakdown,
    MpcObjectiveWeights,
    MpcPlanStep,
    PreheatBlock,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)

RunType = Literal["preheat", "comfort_recovery"]
IntentReplacementPolicy = Literal["keep_active", "replace_if_better"]
IntentFallbackPolicy = Literal["comfort_low_only", "none"]
IntentExecutionMode = Literal[
    "IDLE",
    "PREHEAT_READY",
    "PREHEAT_RUNNING",
    "COMFORT_FALLBACK_RUNNING",
    "LOCKED_OUT",
    "SAFETY_STOP",
]
StartReason = Literal[
    "preheat_intent",
    "comfort_recovery_intent",
    "preheat_intent_comfort_bridge",
    "emergency_comfort_low",
    "external_plant",
    "manual_override",
]
StopReason = Literal[
    "intent_completed",
    "storage_target_reached",
    "post_solar_hold_sufficient",
    "comfort_high_risk",
    "safety_stop",
    "intent_expired",
    "lockout_entered",
    "manual_override",
    "external_plant",
    "missing_or_expired_intent_reset",
]
AuthorityViolationType = Literal[
    "start_without_valid_reason",
    "stop_without_valid_reason",
    "start_without_intent_or_emergency",
    "hp_start_allowed_false_but_start_true",
    "hp_must_be_on_true_but_hp_on_false",
    "hp_must_be_off_true_but_hp_on_true",
    "comfort_fallback_without_comfort_low_risk",
    "solver_created_start",
    "stale_intent_start",
    "missing_or_expired_intent_not_reset",
]


class PreheatRunIntent(DomainModel):
    intent_id: str
    run_type: RunType = "preheat"
    source_block_id: int | None = Field(default=None, ge=0)
    start_window_start_utc: datetime
    start_window_end_utc: datetime
    latest_start_utc: datetime
    planned_start_utc: datetime
    planned_end_utc: datetime
    min_run_steps: int = Field(default=0, ge=0)
    target_charge_kwh: float = Field(default=0.0, ge=0.0)
    target_post_solar_min_temp_c: float = 0.0
    max_preheat_target_c: float = 0.0
    max_starts: int = Field(default=1, ge=0)
    priority: int = 0
    valid_until_utc: datetime
    replacement_policy: IntentReplacementPolicy = "replace_if_better"
    fallback_policy: IntentFallbackPolicy = "comfort_low_only"
    selected: bool = True
    score: float = 0.0
    expected_captured_pv_kwh: float = 0.0
    expected_post_solar_hold_min_temp_c: float = 0.0
    expected_used_charge_kwh: float = 0.0
    expected_later_start_count: int = Field(default=0, ge=0)
    replacement_reason: str | None = None
    keep_reason: str | None = None


class RejectedIntentCandidate(DomainModel):
    source_block_id: int | None = Field(default=None, ge=0)
    reason: str
    score: float = 0.0
    start_window_start_utc: datetime | None = None
    start_window_end_utc: datetime | None = None
    target_charge_kwh: float = 0.0
    expected_captured_pv_kwh: float = 0.0
    expected_post_solar_hold_min_temp_c: float = 0.0
    keep_or_replace_reason: str | None = None


class RunIntentPlan(DomainModel):
    intents: list[PreheatRunIntent] = Field(default_factory=list)
    rejected_candidates: list[RejectedIntentCandidate] = Field(default_factory=list)
    selected_intent_count: int = Field(default=0, ge=0)
    total_target_charge_kwh: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


class RunExecutionState(DomainModel):
    active_intent_id: str | None = None
    active_run_id: str | None = None
    mode: IntentExecutionMode = "IDLE"
    run_started_at_utc: datetime | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    target_charge_kwh: float = Field(default=0.0, ge=0.0)
    used_charge_kwh: float = Field(default=0.0, ge=0.0)
    stop_reason: StopReason | None = None
    start_reason: StartReason | None = None
    previous_hp_on: bool = False
    on_steps: int = Field(default=0, ge=0)
    off_steps: int = Field(default=0, ge=0)
    active_source_block_id: int | None = Field(default=None, ge=0)
    active_intent_started_at_utc: datetime | None = None
    start_stop_ledger: list["StartStopLedgerEntry"] = Field(default_factory=list)
    authority_violations: list["AuthorityViolation"] = Field(default_factory=list)


class RunIntentExecutionTargetStep(DomainModel):
    timestamp_utc: datetime
    active_intent_id: str | None = None
    active_run_id: str | None = None
    eligible_intent_id: str | None = None
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = False
    target_charge_remaining_kwh: float = Field(default=0.0, ge=0.0)
    max_preheat_target_c: float = 0.0
    intent_type: RunType | None = None
    start_reason_hint: StartReason | None = None
    stop_reason_hint: StopReason | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    mode: IntentExecutionMode = "IDLE"
    starts_blocked_no_intent: bool = False
    comfort_fallback_allowed: bool = False
    predicted_min_temp_without_start: float | None = None
    room_temp_c: float | None = None
    mass_temp_c: float | None = None
    q_heat_eff_kw: float = Field(default=0.0, ge=0.0)
    comfort_low_risk: bool = False
    comfort_high_risk: bool = False


class RunIntentPlanningPolicy(DomainModel):
    max_selected_intents: int = Field(default=1, ge=1)
    keep_existing_intent_bonus: float = Field(default=0.75, ge=0.0)
    minimum_improvement_to_replace_intent: float = Field(default=1.0, ge=0.0)
    minimum_time_before_replanning_active_intent_minutes: int = Field(default=30, ge=0)


class IntentPlanningStep(DomainModel):
    index: int = Field(ge=0)
    timestamp_utc: datetime
    temp_min_c: float
    temp_max_c: float
    economic_target_c: float
    room_temp_c: float = 0.0
    mass_temp_c: float | None = None
    q_heat_eff_kw: float = Field(default=0.0, ge=0.0)
    no_heat_room_temp_c: float = 0.0
    no_heat_mass_temp_c: float | None = None
    available_storage_kwh: float = Field(default=0.0, ge=0.0)
    expected_discharge_need_kwh: float = Field(default=0.0, ge=0.0)
    pv_surplus_forecast_kw: float = Field(default=0.0, ge=0.0)
    pv_surplus_window_kwh: float = Field(default=0.0, ge=0.0)
    post_solar_no_heat_min_temp_c: float | None = None
    post_solar_no_heat_end_temp_c: float | None = None
    post_solar_no_heat_drops_below_economic_target: bool = False
    post_solar_no_heat_drops_below_temp_min: bool = False


class IntentPlanningState(DomainModel):
    steps: list[IntentPlanningStep] = Field(default_factory=list)
    total_available_storage_kwh: float = Field(default=0.0, ge=0.0)
    total_expected_discharge_need_kwh: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


class IntentAwareMpcControllerRequest(DomainModel):
    interval_minutes: int = Field(gt=0)
    horizon: list[MpcHorizonStep]
    sequencer_key: str | None = None
    constraints: MpcConstraints = Field(default_factory=MpcConstraints)
    objective_weights: MpcObjectiveWeights = Field(default_factory=MpcObjectiveWeights)
    max_solver_seconds: float | None = Field(default=None, gt=0.0)
    run_execution_state: RunExecutionState | None = None
    previous_intent_plan: RunIntentPlan | None = None
    planning_policy: RunIntentPlanningPolicy = Field(default_factory=RunIntentPlanningPolicy)


class IntentAwareMpcPlan(DomainModel):
    status: str
    termination_condition: str
    feasible: bool
    objective_value: float | None = None
    solve_time_seconds: float | None = None
    heating_explanation: str | None = None
    objective_breakdown: MpcObjectiveBreakdown = Field(default_factory=MpcObjectiveBreakdown)
    steps: list[MpcPlanStep] = Field(default_factory=list)
    intent_planning_state: IntentPlanningState | None = None
    run_intent_plan: RunIntentPlan | None = None
    run_execution_state: RunExecutionState | None = None
    execution_targets: list[RunIntentExecutionTargetStep] = Field(default_factory=list)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)
    start_stop_ledger: list["StartStopLedgerEntry"] = Field(default_factory=list)
    invariant_report: "AuthorityInvariantReport" = Field(
        default_factory=lambda: AuthorityInvariantReport()
    )


class IntentAwareMpcProblem(DomainModel):
    interval_minutes: int = Field(gt=0)
    control_model: Rc2StateThermalControlModel
    initial_state: Rc2StateMpcInitialState | MpcInitialState
    horizon: list[MpcHorizonStep]
    intent_planning_state: IntentPlanningState | None = None
    run_intent_plan: RunIntentPlan | None = None
    execution_targets: list[RunIntentExecutionTargetStep] = Field(default_factory=list)
    constraints: MpcConstraints = Field(default_factory=MpcConstraints)
    objective_weights: MpcObjectiveWeights = Field(default_factory=MpcObjectiveWeights)
    max_solver_seconds: float | None = Field(default=None, gt=0.0)

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0


class StartStopLedgerEntry(DomainModel):
    timestamp: datetime
    transition: Literal["start", "stop"]
    hp_on_previous: bool
    hp_on_current: bool
    start_reason: StartReason | None = None
    stop_reason: StopReason | None = None
    intent_id: str | None = None
    intent_type: RunType | None = None
    active_run_id: str | None = None
    sequencer_mode_before: IntentExecutionMode
    sequencer_mode_after: IntentExecutionMode
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = False
    comfort_low_risk: bool = False
    comfort_high_risk: bool = False
    predicted_min_temp_without_start: float | None = None
    room_temp_c: float | None = None
    mass_temp_c: float | None = None
    q_heat_eff_kw: float = Field(default=0.0, ge=0.0)


class AuthorityViolation(DomainModel):
    timestamp: datetime
    violation: AuthorityViolationType
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AuthorityInvariantReport(DomainModel):
    total_starts: int = 0
    total_stops: int = 0
    starts_by_reason: dict[str, int] = Field(default_factory=dict)
    stops_by_reason: dict[str, int] = Field(default_factory=dict)
    starts_outside_intents: int = 0
    emergency_starts: int = 0
    external_starts: int = 0
    start_stop_violation_count: int = 0
    violation_breakdown: dict[str, int] = Field(default_factory=dict)
