from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from home_optimizer.domain.models import DomainModel
from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
    HeatPumpSequencerState,
    MpcConstraints,
    MpcControllerRequest,
    MpcControlMode,
    MpcHorizonStep,
    MpcObjectiveBreakdown,
    MpcObjectiveWeights,
    MpcPlanStep,
    ThermalFlexibilityState,
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
    stop_reason: str | None = None
    start_reason: str | None = None
    previous_hp_on: bool = False
    on_steps: int = Field(default=0, ge=0)
    off_steps: int = Field(default=0, ge=0)
    active_source_block_id: int | None = Field(default=None, ge=0)
    active_intent_started_at_utc: datetime | None = None


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
    start_reason_hint: str | None = None
    stop_reason_hint: str | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    mode: IntentExecutionMode = "IDLE"
    starts_blocked_no_intent: bool = False
    comfort_fallback_allowed: bool = False


class RunIntentPlanningPolicy(DomainModel):
    max_selected_intents: int = Field(default=1, ge=1)
    keep_existing_intent_bonus: float = Field(default=0.75, ge=0.0)
    minimum_improvement_to_replace_intent: float = Field(default=1.0, ge=0.0)
    minimum_time_before_replanning_active_intent_minutes: int = Field(default=30, ge=0)


class IntentAwareMpcControllerRequest(DomainModel):
    interval_minutes: int = Field(gt=0)
    horizon: list[MpcHorizonStep]
    control_mode: MpcControlMode = "hierarchical_preheat"
    sequencer_key: str | None = None
    constraints: MpcConstraints = Field(default_factory=MpcConstraints)
    objective_weights: MpcObjectiveWeights = Field(default_factory=MpcObjectiveWeights)
    max_solver_seconds: float | None = Field(default=None, gt=0.0)
    run_execution_state: RunExecutionState | None = None
    previous_intent_plan: RunIntentPlan | None = None
    planning_policy: RunIntentPlanningPolicy = Field(default_factory=RunIntentPlanningPolicy)


class IntentAwareMpcPlan(DomainModel):
    control_mode: MpcControlMode = "hierarchical_preheat"
    status: str
    termination_condition: str
    feasible: bool
    objective_value: float | None = None
    solve_time_seconds: float | None = None
    objective_breakdown: MpcObjectiveBreakdown = Field(
        default_factory=MpcObjectiveBreakdown
    )
    steps: list[MpcPlanStep] = Field(default_factory=list)
    thermal_flexibility: ThermalFlexibilityState | None = None
    run_intent_plan: RunIntentPlan | None = None
    run_execution_state: RunExecutionState | None = None
    execution_targets: list[RunIntentExecutionTargetStep] = Field(default_factory=list)
    legacy_execution_targets: list[ExecutionTargetStep] = Field(default_factory=list)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


def to_legacy_sequencer_state(state: RunExecutionState | None) -> HeatPumpSequencerState | None:
    if state is None:
        return None
    return HeatPumpSequencerState(
        mode=(
            "PREHEAT_RUNNING"
            if state.mode == "PREHEAT_RUNNING"
            else "COMFORT_RUNNING"
            if state.mode == "COMFORT_FALLBACK_RUNNING"
            else "LOCKED_OUT"
            if state.mode == "LOCKED_OUT"
            else "SAFETY_STOP"
            if state.mode == "SAFETY_STOP"
            else "IDLE"
        ),
        active_run_id=state.active_run_id,
        active_block_id=state.active_source_block_id,
        run_started_at_utc=state.run_started_at_utc,
        committed_on_until_utc=state.committed_on_until_utc,
        locked_off_until_utc=state.locked_off_until_utc,
        previous_hp_on=state.previous_hp_on,
        on_steps=state.on_steps,
        off_steps=state.off_steps,
        last_start_reason=state.start_reason,
        last_stop_reason=state.stop_reason,
    )


def to_legacy_request(
    request: IntentAwareMpcControllerRequest,
    *,
    horizon: list[MpcHorizonStep],
) -> MpcControllerRequest:
    return MpcControllerRequest(
        interval_minutes=request.interval_minutes,
        horizon=horizon,
        control_mode=request.control_mode,
        sequencer_key=request.sequencer_key,
        sequencer_state=to_legacy_sequencer_state(request.run_execution_state),
        constraints=request.constraints,
        objective_weights=request.objective_weights,
        max_solver_seconds=request.max_solver_seconds,
    )
