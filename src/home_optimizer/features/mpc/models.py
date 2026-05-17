from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import GTI_PV
from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow


class LinearThermalControlModel(DomainModel):
    a: float
    b_out: float
    b_solar: float
    b_heat: float
    b_occ: float
    actuator_alpha: float = Field(default=0.0, ge=0.0, lt=1.0)
    c: float = 0.0
    notes: str = Field(
        default="Control-oriented linear 1-state room temperature model for space-heating MPC."
    )

    def predict_next_temperature(
        self,
        *,
        room_temp_c: float,
        outdoor_temp_c: float,
        solar_gain_kw: float,
        heating_effect_kw: float,
        occupied: float,
    ) -> float:
        return float(
            (self.a * room_temp_c)
            + (self.b_out * outdoor_temp_c)
            + (self.b_solar * solar_gain_kw)
            + (self.b_heat * heating_effect_kw)
            + (self.b_occ * occupied)
            + self.c
        )


class Rc2StateThermalControlModel(DomainModel):
    a11: float
    a12: float
    a21: float
    a22: float
    b_out_room: float
    b_out_mass: float
    b_heat_room: float = 0.0
    b_heat_mass: float
    b_solar_direct_room: float
    b_solar_filtered_room: float = 0.0
    b_solar_direct_mass: float = 0.0
    b_solar_filtered_mass: float = 0.0
    b_occ_room: float
    b_occ_mass: float = 0.0
    b_hour_sin_room: float = 0.0
    b_hour_cos_room: float = 0.0
    b_hour_sin_mass: float = 0.0
    b_hour_cos_mass: float = 0.0
    actuator_alpha: float = Field(default=0.0, ge=0.0, lt=1.0)
    c_room: float = 0.0
    c_mass: float = 0.0
    notes: str = Field(
        default="Control-oriented 2-state room/mass thermal model for space-heating MPC."
    )

    def predict_next_state(
        self,
        *,
        room_temp_c: float,
        mass_temp_c: float,
        outdoor_temp_c: float,
        solar_gain_kw: float,
        solar_gain_mass_kw: float,
        heating_effect_kw: float,
        occupied: float,
        hour_sin: float,
        hour_cos: float,
    ) -> tuple[float, float]:
        next_room_temp_c = float(
            (self.a11 * room_temp_c)
            + (self.a12 * mass_temp_c)
            + (self.b_out_room * outdoor_temp_c)
            + (self.b_heat_room * heating_effect_kw)
            + (self.b_solar_direct_room * solar_gain_kw)
            + (self.b_solar_filtered_room * solar_gain_mass_kw)
            + (self.b_occ_room * occupied)
            + (self.b_hour_sin_room * hour_sin)
            + (self.b_hour_cos_room * hour_cos)
            + self.c_room
        )
        next_mass_temp_c = float(
            (self.a21 * room_temp_c)
            + (self.a22 * mass_temp_c)
            + (self.b_out_mass * outdoor_temp_c)
            + (self.b_heat_mass * heating_effect_kw)
            + (self.b_solar_direct_mass * solar_gain_kw)
            + (self.b_solar_filtered_mass * solar_gain_mass_kw)
            + (self.b_occ_mass * occupied)
            + (self.b_hour_sin_mass * hour_sin)
            + (self.b_hour_cos_mass * hour_cos)
            + self.c_mass
        )
        return next_room_temp_c, next_mass_temp_c


class MpcHorizonStep(DomainModel):
    timestamp_utc: datetime
    outdoor_temp_c: float
    solar_gain_kw: float = 0.0
    solar_gain_mass_kw: float | None = None
    solar_irradiance_forecast_w_m2: float = 0.0
    solar_irradiance_realized_w_m2: float = 0.0
    solar_gain_realized_kw: float | None = None
    effective_heating_kw_forecast: float = Field(ge=0.0)
    hp_electric_power_forecast_kw: float = Field(default=0.0, ge=0.0)
    pv_available_power_forecast_kw: float = Field(default=0.0, ge=0.0)
    pv_available_power_realized_kw: float | None = None
    base_load_power_forecast_kw: float = Field(default=0.0, ge=0.0)
    base_load_power_realized_kw: float | None = None
    occupied: float = Field(default=0.0, ge=0.0, le=1.0)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    target_temp_c: float | None = None
    temp_min_c: float
    temp_max_c: float
    price_eur_kwh: float = 0.0
    import_price_eur_kwh: float = 0.0
    export_price_eur_kwh: float = 0.0
    realized_room_temp_c: float | None = None
    economic_target_c: float | None = None
    preheat_active: bool = False
    preheat_opportunity_score: float = Field(default=0.0, ge=0.0)
    max_preheat_target_c: float | None = None
    preheat_budget_share_kwh: float = Field(default=0.0, ge=0.0)
    preheat_block_id: int | None = None
    preheat_block_budget_kwh: float = Field(default=0.0, ge=0.0)
    preheat_block_cumulative_target_kwh: float = Field(default=0.0, ge=0.0)
    preheat_block_max_starts: int = Field(default=0, ge=0)
    sequencer_mode: str = "IDLE"
    active_run_id: str | None = None
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = True
    start_reason_hint: str | None = None
    stop_reason_hint: str | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    starts_used_in_block: int = Field(default=0, ge=0)
    run_budget_used_kwh: float = Field(default=0.0, ge=0.0)
    starts_blocked_by_lockout: bool = False
    starts_blocked_by_max_starts: bool = False
    starts_blocked_by_existing_commitment: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> "MpcHorizonStep":
        if self.temp_min_c > self.temp_max_c:
            raise ValueError("temp_min_c cannot be greater than temp_max_c")
        if (
            self.hp_electric_power_forecast_kw == 0.0
            and self.effective_heating_kw_forecast > 0.0
        ):
            object.__setattr__(
                self,
                "hp_electric_power_forecast_kw",
                self.effective_heating_kw_forecast,
            )
        if self.import_price_eur_kwh == 0.0 and self.price_eur_kwh > 0.0:
            object.__setattr__(self, "import_price_eur_kwh", self.price_eur_kwh)
        if self.solar_gain_mass_kw is None:
            object.__setattr__(self, "solar_gain_mass_kw", self.solar_gain_kw)
        if self.solar_gain_realized_kw is None:
            object.__setattr__(self, "solar_gain_realized_kw", self.solar_gain_kw)
        if self.pv_available_power_realized_kw is None:
            object.__setattr__(
                self,
                "pv_available_power_realized_kw",
                self.pv_available_power_forecast_kw,
            )
        if self.base_load_power_realized_kw is None:
            object.__setattr__(
                self,
                "base_load_power_realized_kw",
                self.base_load_power_forecast_kw,
            )
        if self.target_temp_c is None:
            object.__setattr__(self, "target_temp_c", (self.temp_min_c + self.temp_max_c) / 2.0)
        if self.economic_target_c is None:
            object.__setattr__(self, "economic_target_c", float(self.target_temp_c))
        if self.max_preheat_target_c is None:
            object.__setattr__(self, "max_preheat_target_c", float(self.economic_target_c))
        return self


class MpcInitialState(DomainModel):
    room_temp_c: float
    q_heat_eff_kw: float = Field(default=0.0, ge=0.0)
    hp_on: bool = False
    on_steps: int = Field(default=0, ge=0)
    off_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_counters(self) -> "MpcInitialState":
        if self.hp_on and self.off_steps != 0:
            raise ValueError("off_steps must be zero when hp_on is true")
        if not self.hp_on and self.on_steps != 0:
            raise ValueError("on_steps must be zero when hp_on is false")
        return self


class Rc2StateMpcInitialState(MpcInitialState):
    mass_temp_c: float


class MpcConstraints(DomainModel):
    min_on_steps: int = Field(default=0, ge=0)
    min_off_steps: int = Field(default=0, ge=0)
    pv_opportunity_window_steps: int = Field(default=0, ge=0)


class PreheatPlanStep(DomainModel):
    timestamp_utc: datetime
    economic_target_c: float = 0.0
    preheat_active: bool = False
    preheat_opportunity_score: float = Field(default=0.0, ge=0.0)
    max_preheat_target_c: float = 0.0
    preheat_budget_share_kwh: float = Field(default=0.0, ge=0.0)
    preheat_block_id: int | None = None
    preheat_block_budget_kwh: float = Field(default=0.0, ge=0.0)
    preheat_block_max_starts: int = Field(default=0, ge=0)
    pv_surplus_window_kwh: float = Field(default=0.0, ge=0.0)
    storage_headroom_electric_kwh: float = Field(default=0.0, ge=0.0)
    reason: str | None = None


class PreheatBlock(DomainModel):
    block_id: int = Field(ge=0)
    candidate_block_id: int | None = Field(default=None, ge=0)
    selected: bool = True
    skipped_reason: str | None = None
    start_index: int = Field(default=0, ge=0)
    end_index: int = Field(default=0, ge=0)
    start_time_utc: datetime
    end_time_utc: datetime
    available_surplus_kwh: float = Field(default=0.0, ge=0.0)
    available_storage_kwh: float = Field(default=0.0, ge=0.0)
    planned_charge_kwh: float = Field(default=0.0, ge=0.0)
    max_starts: int = Field(default=1, ge=0)
    min_run_steps: int = Field(default=0, ge=0)
    preferred_start_index: int | None = None
    max_preheat_target_c: float = 0.0
    simulated_end_room_temp_c: float | None = None
    simulated_end_mass_temp_c: float | None = None
    post_solar_no_heat_min_temp_c: float | None = None
    post_solar_no_heat_end_temp_c: float | None = None
    post_solar_no_heat_drops_below_economic_target: bool = False
    post_solar_no_heat_drops_below_temp_min: bool = False
    step_count: int = Field(default=0, ge=0)
    planned_run_steps: int = Field(default=0, ge=0)
    used_charge_kwh: float = Field(default=0.0, ge=0.0)
    missed_charge_kwh: float = Field(default=0.0, ge=0.0)
    remaining_need_kwh: float = Field(default=0.0, ge=0.0)
    starts_in_block: int = Field(default=0, ge=0)
    run_duration_minutes: float = Field(default=0.0, ge=0.0)
    limit_reason: str | None = None
    reason: str | None = None


class PreheatPlan(DomainModel):
    steps: list[PreheatPlanStep] = Field(default_factory=list)
    blocks: list[PreheatBlock] = Field(default_factory=list)
    preheat_budget_electric_kwh: float = Field(default=0.0, ge=0.0)
    preheat_window_start_utc: datetime | None = None
    preheat_window_end_utc: datetime | None = None
    reason: str | None = None

    @property
    def preheat_active_per_step(self) -> list[bool]:
        return [step.preheat_active for step in self.steps]

    @property
    def max_preheat_target_c_per_step(self) -> list[float]:
        return [step.max_preheat_target_c for step in self.steps]


class ThermalFlexibilityStep(DomainModel):
    index: int = Field(ge=0)
    timestamp_utc: datetime
    temp_min_c: float
    temp_max_c: float
    economic_target_c: float
    room_temp_c: float = 0.0
    mass_temp_c: float | None = None
    q_heat_eff_kw: float = Field(default=0.0, ge=0.0)
    no_heat_room_temp_c: float
    no_heat_mass_temp_c: float | None = None
    room_mass_delta_c: float = 0.0
    mass_deficit_to_economic_target_c: float = Field(default=0.0, ge=0.0)
    mass_deficit_to_preheat_target_c: float = Field(default=0.0, ge=0.0)
    normalized_storage_soc: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_storage_soc_kwh: float | None = None
    comfort_headroom_c: float = Field(default=0.0, ge=0.0)
    available_storage_kwh: float = Field(default=0.0, ge=0.0)
    expected_discharge_need_kwh: float = Field(default=0.0, ge=0.0)
    pv_surplus_forecast_kw: float = Field(default=0.0, ge=0.0)
    pv_surplus_window_kwh: float = Field(default=0.0, ge=0.0)
    post_solar_no_heat_min_temp_c: float | None = None
    post_solar_no_heat_end_temp_c: float | None = None
    post_solar_no_heat_drops_below_economic_target: bool = False
    post_solar_no_heat_drops_below_temp_min: bool = False


class ThermalFlexibilityState(DomainModel):
    steps: list[ThermalFlexibilityStep] = Field(default_factory=list)
    total_available_storage_kwh: float = Field(default=0.0, ge=0.0)
    total_expected_discharge_need_kwh: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


class PreheatSchedule(DomainModel):
    blocks: list[PreheatBlock] = Field(default_factory=list)
    step_to_block_id: list[int | None] = Field(default_factory=list)
    total_planned_charge_kwh: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


class ExecutionTargetStep(DomainModel):
    timestamp_utc: datetime
    economic_target_c: float
    preheat_target_c: float
    active_preheat_block_id: int | None = None
    remaining_block_budget_kwh: float = Field(default=0.0, ge=0.0)
    block_budget_share_kwh: float = Field(default=0.0, ge=0.0)
    block_cumulative_budget_target_kwh: float = Field(default=0.0, ge=0.0)
    storage_target_kwh: float = Field(default=0.0, ge=0.0)
    max_preheat_target_c: float
    start_allowed_for_preheat: bool = False
    start_reason_hint: str | None = None
    sequencer_mode: str = "IDLE"
    active_run_id: str | None = None
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = True
    stop_reason_hint: str | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    starts_used_in_block: int = Field(default=0, ge=0)
    run_budget_used_kwh: float = Field(default=0.0, ge=0.0)
    starts_blocked_by_lockout: bool = False
    starts_blocked_by_max_starts: bool = False
    starts_blocked_by_existing_commitment: bool = False
    stop_conditions: list[str] = Field(default_factory=list)


MpcControlMode = Literal["hierarchical_preheat"]

HeatPumpSequencerMode = Literal[
    "IDLE",
    "PREHEAT_RUNNING",
    "COMFORT_RUNNING",
    "LOCKED_OUT",
    "SAFETY_STOP",
]


class HeatPumpSequencerState(DomainModel):
    mode: HeatPumpSequencerMode = "IDLE"
    active_run_id: str | None = None
    active_block_id: int | None = None
    run_started_at_utc: datetime | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    starts_used_by_block: dict[int, int] = Field(default_factory=dict)
    used_budget_by_block_kwh: dict[int, float] = Field(default_factory=dict)
    previous_hp_on: bool = False
    on_steps: int = Field(default=0, ge=0)
    off_steps: int = Field(default=0, ge=0)
    last_start_reason: str | None = None
    last_stop_reason: str | None = None


class HeatPumpSequencerSnapshot(DomainModel):
    mode: HeatPumpSequencerMode
    active_run_id: str | None = None
    active_block_id: int | None = None
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = True
    start_reason: str | None = None
    stop_reason: str | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    starts_used_in_block: int = Field(default=0, ge=0)
    run_budget_used_kwh: float = Field(default=0.0, ge=0.0)
    starts_blocked_by_lockout: bool = False
    starts_blocked_by_max_starts: bool = False
    starts_blocked_by_existing_commitment: bool = False
    run_target_budget_kwh: float = Field(default=0.0, ge=0.0)
    stop_conditions: list[str] = Field(default_factory=list)


class MpcObjectiveWeights(DomainModel):
    comfort_low: float = Field(default=10_000.0, ge=0.0)
    comfort_high: float = Field(default=10_000.0, ge=0.0)
    active_comfort_high: float | None = Field(default=None, ge=0.0)
    passive_comfort_high: float = Field(default=100.0, ge=0.0)
    tracking_under_target: float = Field(default=25.0, ge=0.0)
    tracking_over_target: float = Field(default=0.5, ge=0.0)
    unnecessary_heating: float = Field(default=4.0, ge=0.0)
    q_heat_eff_active_threshold_kw: float = Field(default=0.1, ge=0.0)
    terminal: float = Field(default=8.0, ge=0.0)
    start: float = Field(default=10.0, ge=0.0)
    energy: float = Field(default=1.0, ge=0.0)
    pv_self_consumption: float = Field(default=12.0, ge=0.0)
    preheat_budget_shortfall: float = Field(default=24.0, ge=0.0)
    runtime: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _resolve_active_comfort_high(self) -> "MpcObjectiveWeights":
        if self.active_comfort_high is None:
            object.__setattr__(self, "active_comfort_high", self.comfort_high)
        return self


class MpcProblem(DomainModel):
    interval_minutes: int = Field(gt=0)
    control_mode: MpcControlMode = "hierarchical_preheat"
    control_model: Rc2StateThermalControlModel | LinearThermalControlModel
    initial_state: Rc2StateMpcInitialState | MpcInitialState
    horizon: list[MpcHorizonStep]
    preheat_plan: PreheatPlan | None = None
    thermal_flexibility: ThermalFlexibilityState | None = None
    preheat_schedule: PreheatSchedule | None = None
    execution_targets: list[ExecutionTargetStep] | None = None
    sequencer_state: HeatPumpSequencerState | None = None
    constraints: MpcConstraints = Field(default_factory=MpcConstraints)
    objective_weights: MpcObjectiveWeights = Field(default_factory=MpcObjectiveWeights)
    max_solver_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _validate_horizon(self) -> "MpcProblem":
        if not self.horizon:
            raise ValueError("horizon cannot be empty")
        timestamps = [step.timestamp_utc for step in self.horizon]
        if timestamps != sorted(timestamps):
            raise ValueError("horizon timestamps must be sorted ascending")
        if isinstance(self.control_model, Rc2StateThermalControlModel) and not isinstance(
            self.initial_state,
            Rc2StateMpcInitialState,
        ):
            raise ValueError("Rc2StateThermalControlModel requires Rc2StateMpcInitialState")
        return self

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0


class MpcPlanStep(DomainModel):
    timestamp_utc: datetime
    hp_on: bool
    start: bool
    stop: bool
    predicted_room_temp_c: float
    economic_target_c: float = 0.0
    useful_preheat_target_c: float = 0.0
    preheat_active: bool = False
    preheat_opportunity_score: float = 0.0
    preheat_budget_share_kwh: float = 0.0
    preheat_charge_kwh: float = 0.0
    preheat_block_id: int | None = None
    preheat_block_budget_kwh: float = 0.0
    mass_temp_c: float | None = None
    q_heat_eff_kw: float = 0.0
    sequencer_mode: str = "IDLE"
    active_run_id: str | None = None
    hp_must_be_on: bool = False
    hp_must_be_off: bool = False
    hp_start_allowed: bool = True
    start_reason: str | None = None
    stop_reason: str | None = None
    committed_on_until_utc: datetime | None = None
    locked_off_until_utc: datetime | None = None
    starts_used_in_block: int = 0
    run_budget_used_kwh: float = 0.0
    starts_blocked_by_lockout: bool = False
    starts_blocked_by_max_starts: bool = False
    starts_blocked_by_existing_commitment: bool = False
    temp_min_c: float
    temp_max_c: float
    slack_low_c: float = 0.0
    slack_high_c: float = 0.0
    effective_heating_kw: float = 0.0
    price_eur_kwh: float = 0.0
    estimated_energy_cost_eur: float = 0.0


class MpcObjectiveBreakdown(DomainModel):
    comfort_low: float = 0.0
    active_comfort_high: float = 0.0
    passive_comfort_high: float = 0.0
    tracking_under_target: float = 0.0
    tracking_over_target: float = 0.0
    unnecessary_heating: float = 0.0
    terminal: float = 0.0
    start: float = 0.0
    runtime: float = 0.0
    energy_cost: float = 0.0
    pv_self_consumption_reward: float = 0.0
    captured_pv_kwh: float = 0.0
    preheat_budget_shortfall: float = 0.0

    @property
    def comfort_high(self) -> float:
        return self.active_comfort_high + self.passive_comfort_high

    @property
    def comfort_total(self) -> float:
        return self.comfort_low + self.comfort_high

    @property
    def temperature_tracking(self) -> float:
        return self.tracking_under_target + self.tracking_over_target

    @property
    def energy(self) -> float:
        return self.energy_cost

    @property
    def total(self) -> float:
        return (
            self.comfort_low
            + self.comfort_high
            + self.tracking_under_target
            + self.tracking_over_target
            + self.unnecessary_heating
            + self.terminal
            + self.start
            + self.runtime
            + self.energy_cost
            + self.preheat_budget_shortfall
            - self.pv_self_consumption_reward
        )


class MpcPlan(DomainModel):
    control_mode: MpcControlMode = "hierarchical_preheat"
    status: str
    termination_condition: str
    feasible: bool
    objective_value: float | None = None
    solve_time_seconds: float | None = None
    heating_explanation: str | None = None
    preheat_plan: PreheatPlan | None = None
    thermal_flexibility: ThermalFlexibilityState | None = None
    preheat_schedule: PreheatSchedule | None = None
    sequencer_state: HeatPumpSequencerState | None = None
    objective_breakdown: MpcObjectiveBreakdown = Field(
        default_factory=MpcObjectiveBreakdown
    )
    steps: list[MpcPlanStep] = Field(default_factory=list)


class MpcControllerRequest(DomainModel):
    interval_minutes: int = Field(gt=0)
    horizon: list[MpcHorizonStep]
    control_mode: MpcControlMode = "hierarchical_preheat"
    preheat_plan: PreheatPlan | None = None
    sequencer_state: HeatPumpSequencerState | None = None
    sequencer_key: str | None = None
    constraints: MpcConstraints = Field(default_factory=MpcConstraints)
    objective_weights: MpcObjectiveWeights = Field(default_factory=MpcObjectiveWeights)
    max_solver_seconds: float | None = Field(default=None, gt=0.0)


class ControlModelConversionOptions(DomainModel):
    solar_gain_input_scale: float = Field(default=1.0, gt=0.0)


class MpcHorizonBuildRequest(DomainModel):
    start_time_utc: datetime
    horizon_steps: int = Field(gt=0)
    interval_minutes: int = Field(gt=0)
    target_schedule: list[TemperatureTargetWindow]
    forecast_entries: list[ForecastEntry] = Field(default_factory=list)
    price_intervals: list[PriceInterval] = Field(default_factory=list)
    default_effective_heating_kw: float = Field(ge=0.0)
    outdoor_temperature_name: str = "temperature"
    solar_gain_name: str = "gti_living_room_windows_adjusted"
    pv_power_name: str = GTI_PV
    solar_gain_input_scale: float = Field(default=1.0, gt=0.0)
    solar_gain_filter_alpha: float = Field(default=0.0, ge=0.0, lt=1.0)
    initial_filtered_solar_gain_kw: float = 0.0
    pv_power_input_scale: float = Field(default=0.0, ge=0.0)
    default_hp_electric_power_kw: float = Field(default=0.0, ge=0.0)
    default_base_load_power_kw: float = Field(default=0.0)
    default_export_price_eur_kwh: float = Field(default=0.0, ge=0.0)
    default_occupied: float = Field(default=0.0, ge=0.0, le=1.0)
    local_timezone: str | None = None
    fallback_temp_min_c: float | None = None
    fallback_temp_max_c: float | None = None

    @model_validator(mode="after")
    def _validate_fallback_bounds(self) -> "MpcHorizonBuildRequest":
        if (
            self.fallback_temp_min_c is not None
            and self.fallback_temp_max_c is not None
            and self.fallback_temp_min_c > self.fallback_temp_max_c
        ):
            raise ValueError("fallback_temp_min_c cannot be greater than fallback_temp_max_c")
        return self
