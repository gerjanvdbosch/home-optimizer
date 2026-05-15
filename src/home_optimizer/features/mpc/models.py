from __future__ import annotations

from datetime import datetime
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
    effective_heating_kw_forecast: float = Field(ge=0.0)
    hp_electric_power_forecast_kw: float = Field(default=0.0, ge=0.0)
    pv_available_power_forecast_kw: float = Field(default=0.0, ge=0.0)
    base_load_power_forecast_kw: float = Field(default=0.0, ge=0.0)
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
        if self.target_temp_c is None:
            object.__setattr__(self, "target_temp_c", (self.temp_min_c + self.temp_max_c) / 2.0)
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
    min_on_steps: int = Field(default=1, ge=1)
    min_off_steps: int = Field(default=1, ge=1)


class MpcObjectiveWeights(DomainModel):
    comfort_low: float = Field(default=10_000.0, ge=0.0)
    comfort_high: float = Field(default=10_000.0, ge=0.0)
    temperature_tracking: float = Field(default=5.0, ge=0.0)
    start: float = Field(default=250.0, ge=0.0)
    energy: float = Field(default=1.0, ge=0.0)
    runtime: float = Field(default=0.1, ge=0.0)


class MpcProblem(DomainModel):
    interval_minutes: int = Field(gt=0)
    control_model: Rc2StateThermalControlModel | LinearThermalControlModel
    initial_state: Rc2StateMpcInitialState | MpcInitialState
    horizon: list[MpcHorizonStep]
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
    q_heat_eff_kw: float = 0.0
    temp_min_c: float
    temp_max_c: float
    slack_low_c: float = 0.0
    slack_high_c: float = 0.0
    effective_heating_kw: float = 0.0
    price_eur_kwh: float = 0.0
    estimated_energy_cost_eur: float = 0.0


class MpcObjectiveBreakdown(DomainModel):
    comfort_low: float = 0.0
    comfort_high: float = 0.0
    temperature_tracking: float = 0.0
    terminal: float = 0.0
    start: float = 0.0
    runtime: float = 0.0
    energy: float = 0.0

    @property
    def comfort_total(self) -> float:
        return self.comfort_low + self.comfort_high

    @property
    def total(self) -> float:
        return (
            self.comfort_low
            + self.comfort_high
            + self.temperature_tracking
            + self.terminal
            + self.start
            + self.runtime
            + self.energy
        )


class MpcPlan(DomainModel):
    status: str
    termination_condition: str
    feasible: bool
    objective_value: float | None = None
    solve_time_seconds: float | None = None
    heating_explanation: str | None = None
    objective_breakdown: MpcObjectiveBreakdown = Field(
        default_factory=MpcObjectiveBreakdown
    )
    steps: list[MpcPlanStep] = Field(default_factory=list)


class MpcControllerRequest(DomainModel):
    interval_minutes: int = Field(gt=0)
    horizon: list[MpcHorizonStep]
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
