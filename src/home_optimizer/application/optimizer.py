"""Core MPC optimisation logic — the ``Optimizer`` class.

Architecture
------------
This module contains the **domain core** of the Home Optimizer.  It is
deliberately kept free of any web-framework or visualisation concerns; those
belong in ``api.py`` (FastAPI).  Periodic APScheduler orchestration is exposed
via methods on :class:`Optimizer` so API and background execution share one
validated path.

The public surface is intentionally small:

* :class:`MPCStepResult` — an immutable dataclass carrying all numerical
  outputs of one optimisation step (control sequences, COP arrays, energy
  summaries, forecasts).  No charts — chart generation is the API's concern.
* :class:`Optimizer` — a stateless service object that accepts a
  :class:`RunRequest`, orchestrates the full solve
  pipeline (thermal model → COP model → forecasts → MPC → summaries) and
  returns an :class:`MPCStepResult`.

Design decisions
----------------
* **No magic numbers** — every numeric constant originates from the
  validated ``RunRequest`` (Pydantic), never from literals here.
* **Fail-fast** — invalid physics (non-positive capacities, COP ≤ 1, etc.)
  are caught by Pydantic and the ``HeatPumpCOPModel`` *before* the CVXPY
  solver is invoked.
* **DRY** — both the HTTP endpoint (``api.py``) and periodic scheduler calls
  use :meth:`Optimizer.solve`, not a copy of the logic.

Units: power [kW], temperature [°C], energy [kWh], time [h].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from .forecasting import (
    ForecastBuilder,
    _FORECAST_SERVICE,
    build_repository_forecast_overrides,
    inject_forecast_overrides,
)
from .pipeline import OptimizerPipeline
from ..control.mpc import MPCController, MPCSolution
from ..domain.dhw.model import DHWModel
from ..domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters
from ..domain.ufh.model import ThermalModel
from ..pricing import BasePriceModel, PriceConfig, build_price_model  # noqa: F401 — PriceConfig re-exported via RunRequest
from ..types.calibration import CalibrationParameterOverrides
from ..types.constants import LAMBDA_WATER_KWH_PER_M3_K, W_PER_KW
from ..types.control import CombinedMPCParameters, DHWMPCParameters, MPCParameters
from ..types.forecast import DHWForecastHorizon, ForecastHorizon
from ..types.physical import DHWParameters, ThermalParameters

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

    from ..sensors.base import SensorBackend
    from ..telemetry.repository import TelemetryRepository

log = logging.getLogger("home_optimizer.application.optimizer")

# ---------------------------------------------------------------------------
# Shared request model (API + scheduler + local runners)
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """All user-adjustable parameters for one MPC optimisation step.

    The model uses Pydantic validation to enforce physical bounds on every
    parameter before they reach the solver.
    """

    # ── UFH: two-zone house thermal model (§3–§5) ─────────────────────────
    C_r: float = Field(
        6.0, ge=0.5, le=50.0, description="Room-air + furniture thermal capacity C_r [kWh/K]"
    )
    C_b: float = Field(
        10.0, ge=1.0, le=200.0, description="UFH floor / concrete slab thermal capacity C_b [kWh/K]"
    )
    R_br: float = Field(
        1.0, ge=0.1, le=20.0, description="Thermal resistance floor -> room R_br [K/kW]"
    )
    R_ro: float = Field(
        10.0, ge=0.1, le=30.0, description="Thermal resistance room -> outside R_ro [K/kW]"
    )
    alpha: float = Field(
        0.25, ge=0.0, le=1.0, description="Fraction of solar gain to room air alpha [-]"
    )
    eta: float = Field(0.55, ge=0.0, le=1.0, description="Window solar transmittance eta [-]")
    A_glass: float = Field(
        7.5, ge=0.5, le=40.0, description="South-facing glazing area A_glass [m^2]"
    )

    # ── UFH: initial conditions ───────────────────────────────────────────
    T_r_init: float = Field(
        20.5, ge=5.0, le=35.0, description="Initial room-air temperature T_r [degC]"
    )
    T_b_init: float = Field(
        22.5, ge=5.0, le=45.0, description="Initial floor/slab temperature T_b [degC]"
    )
    room_temperature_bias_c: float = Field(
        0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Additive room-temperature sensor bias correction [°C]. "
            "The corrected room reading equals raw_sensor + room_temperature_bias_c."
        ),
    )
    previous_power_kw: float = Field(
        0.8, ge=0.0, le=20.0, description="UFH power applied in previous step [kW]"
    )

    # ── MPC settings (§14) ────────────────────────────────────────────────
    horizon_hours: int = Field(24, ge=4, le=48, description="Horizon length N [steps]")
    dt_hours: float = Field(1.0, ge=0.25, le=2.0, description="Forward-Euler time step dt [h]")
    Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c [dimensionless]")
    R_c: float = Field(0.05, ge=0.0, description="Regularisation weight R_c")
    Q_N: float = Field(12.0, ge=0.0, description="Terminal comfort weight Q_N")
    P_max: float = Field(4.5, ge=0.5, le=20.0, description="Max UFH thermal power P_UFH,max [kW]")
    delta_P_max: float = Field(
        1.0, ge=0.1, le=10.0, description="Max UFH ramp-rate delta_P_UFH,max [kW/step]"
    )
    T_min: float = Field(
        19.0, ge=10.0, le=25.0, description="Minimum comfort temperature T_min [degC]"
    )
    T_max: float = Field(
        22.5, ge=16.0, le=30.0, description="Maximum comfort temperature T_max [degC]"
    )
    T_ref: float = Field(20.5, ge=15.0, le=26.0, description="Comfort setpoint T_ref [degC]")

    # ── UFH: disturbance forecast ─────────────────────────────────────────
    outdoor_temperature_c: float = Field(
        6.0, ge=-20.0, le=35.0, description="Outdoor temperature T_out [degC] (scalar fallback)"
    )
    t_out_forecast: list[float] | None = Field(
        None,
        description=(
            "Hourly outdoor temperature forecast [°C], length N.  "
            "When provided (from Open-Meteo via ForecastPersister) this array "
            "overrides the scalar outdoor_temperature_c for every step of the horizon."
        ),
    )
    gti_window_forecast: list[float] | None = Field(
        None,
        description=(
            "Hourly GTI forecast for south-facing windows [W/m²], length N.  "
            "Must be provided via ForecastPersister (Open-Meteo).  "
            "Raises ValueError when absent."
        ),
    )
    shutter_living_room_pct: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description=(
            "Living-room shutter opening [%]. 100 = fully open, 0 = fully closed. "
            "The UFH solar-gain disturbance uses η_eff = η × (shutter / 100)."
        ),
    )
    shutter_forecast: list[float] | None = Field(
        None,
        description=(
            "Living-room shutter opening forecast [%], length N.  "
            "When provided, this array overrides the scalar shutter_living_room_pct "
            "for every MPC step.  100 = fully open, 0 = fully closed."
        ),
    )
    gti_pv_forecast: list[float] | None = Field(
        None,
        description=(
            "Hourly GTI forecast for PV panels [W/m²], length N.  "
            "PV power is derived as (gti_pv / W_PER_KW) * pv_peak_kw.  "
            "Raises ValueError when absent and pv_enabled=True."
        ),
    )
    price_config: PriceConfig = Field(
        default_factory=PriceConfig,
        description=(
            "Electricity price model: flat rate, dual-tariff (piek/dal + "
            "terugleververgoeding), or real Nordpool day-ahead prices.  "
            "See PriceConfig for all sub-fields."
        ),
    )
    baseload_forecast: list[float] | None = Field(
        None,
        description=(
            "Forecast household baseload / non-heat-pump electrical demand [kW], length N.  "
            "This signal is learned from telemetry and can be mapped to time-varying "
            "internal gains when internal_gains_forecast is absent."
        ),
    )
    internal_gains_kw: float = Field(
        0.30,
        ge=0.0,
        le=3.0,
        description=(
            "Baseline internal heat gains Q_int [kW].  Used directly when no horizon-wide "
            "internal-gains or baseload forecast is available, and otherwise acts as the "
            "background non-electrical gain offset (occupants, standby heat, etc.)."
        ),
    )
    internal_gains_forecast: list[float] | None = Field(
        None,
        description=(
            "Forecast internal gains Q_int [kW], length N.  "
            "When provided, this array overrides the scalar internal_gains_kw."
        ),
    )
    internal_gains_heat_fraction: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the electrical household baseload forecast that becomes useful indoor "
            "heat gain [-].  Used only when baseload_forecast is available and explicit "
            "internal_gains_forecast is absent."
        ),
    )

    # ── PV self-consumption ───────────────────────────────────────────────
    pv_enabled: bool = Field(
        True, description="Enable PV self-consumption (reduces net grid cost)"
    )
    pv_peak_kw: float = Field(2.0, ge=0.0, le=20.0, description="PV system peak capacity [kW]")

    # ── DHW: two-node stratification tank (§7–§11) ───────────────────────
    dhw_enabled: bool = Field(True, description="Enable DHW (domestic hot water) control")
    dhw_C_top: float = Field(
        0.5814, ge=0.01, le=5.0, description="DHW top-layer thermal capacity C_top [kWh/K]"
    )
    dhw_C_bot: float = Field(
        0.5814, ge=0.01, le=5.0, description="DHW bottom-layer thermal capacity C_bot [kWh/K]"
    )
    dhw_R_strat: float = Field(
        10.0,
        gt=0.0,
        le=100.0,
        description=(
            "Effective DHW stratification resistance R_strat [K/kW]. Positive values very "
            "close to zero are physically admissible and represent near-perfect mixing during charging."
        ),
    )
    dhw_R_loss: float = Field(
        50.0,
        ge=5.0,
        description=(
            "Standby-loss resistance R_loss [K/kW]. High-efficiency tanks can exceed "
            "older heuristic ceilings, so only a physical lower bound is enforced."
        ),
    )
    dhw_lambda_water_kwh_per_m3k: float = Field(
        LAMBDA_WATER_KWH_PER_M3_K,
        gt=0.0,
        description="Water volumetric heat capacity lambda [kWh/(m^3·K)] (§8.4)",
    )
    dhw_T_top_init: float = Field(
        55.0, ge=20.0, le=85.0, description="Initial top-layer temperature T_top [degC]"
    )
    dhw_T_bot_init: float = Field(
        45.0, ge=15.0, le=80.0, description="Initial bottom-layer temperature T_bot [degC]"
    )
    dhw_top_temperature_bias_c: float = Field(
        0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Additive DHW top-temperature sensor bias correction [°C]. "
            "The corrected top-layer reading equals raw_sensor + dhw_top_temperature_bias_c."
        ),
    )
    dhw_bottom_temperature_bias_c: float = Field(
        0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Additive DHW bottom-temperature sensor bias correction [°C]. "
            "The corrected bottom-layer reading equals raw_sensor + dhw_bottom_temperature_bias_c."
        ),
    )
    dhw_boiler_ambient_bias_c: float = Field(
        0.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Additive DHW boiler-ambient sensor bias correction [°C]. "
            "The corrected ambient reading equals raw_sensor + dhw_boiler_ambient_bias_c."
        ),
    )
    dhw_P_max: float = Field(
        3.0, ge=0.5, le=15.0, description="Max DHW thermal power P_dhw,max [kW]"
    )
    dhw_delta_P_max: float = Field(
        1.0, ge=0.1, le=10.0, description="Max DHW ramp-rate delta_P_dhw,max [kW/step]"
    )
    dhw_T_min: float = Field(
        50.0, ge=10.0, le=70.0, description="Minimum tap (top-layer) temperature T_dhw,min [degC]"
    )
    dhw_T_legionella: float = Field(
        60.0, ge=55.0, le=85.0, description="Legionella prevention temperature T_leg [degC]"
    )
    dhw_legionella_period_steps: int = Field(
        168, ge=24, le=336, description="Legionella cycle period n_leg [steps]"
    )
    dhw_legionella_duration_steps: int = Field(
        1, ge=1, le=4, description="Min consecutive steps at T_legionella for legionella kill"
    )
    dhw_v_tap_forecast: list[float] | None = Field(
        None,
        description=(
            "Hourly DHW tap-flow forecast Vdot_tap [m³/h], length N. "
            "Provided by the DHW tap forecaster once enough history is available. "
            "When absent the MPC assumes zero tap demand (conservative cold-start). "
            "Can also be set explicitly for simulation or testing."
        ),
    )
    dhw_t_mains_c: float = Field(
        10.0, ge=0.0, le=25.0, description="Cold mains-water temperature T_mains [degC]"
    )
    dhw_t_amb_c: float = Field(
        20.0, ge=5.0, le=35.0, description="Ambient temperature around the boiler T_amb [degC]"
    )

    # ── Shared heat-pump electrical budget ───────────────────────────────
    P_hp_max_elec: float = Field(
        2.5,
        ge=0.5,
        le=30.0,
        description=(
            "Shared heat-pump electrical power budget P_hp,max,elec [kW]. "
            "Enforces P_UFH/COP_UFH + P_dhw/COP_dhw <= P_hp_max_elec (section 14)."
        ),
    )

    # ── Warmtepomp — Carnot COP model (§14.1) ────────────────────────────
    eta_carnot: float = Field(0.45, ge=0.1, le=0.99, description="Carnot efficiency factor eta [-]")
    delta_T_cond: float = Field(
        5.0, ge=0.0, le=15.0, description="Condensing approach temperature delta_cond [K]"
    )
    delta_T_evap: float = Field(
        5.0, ge=0.0, le=15.0, description="Evaporating approach temperature delta_evap [K]"
    )
    T_supply_min: float = Field(
        28.0, ge=15.0, le=60.0, description="Minimum UFH supply temperature T_supply,min [degC]"
    )
    T_ref_outdoor_curve: float = Field(
        18.0,
        ge=5.0,
        le=25.0,
        description="Balance-point outdoor temperature for heating curve [degC]",
    )
    heating_curve_slope: float = Field(1.0, ge=0.0, le=3.0, description="Heating curve slope [K/K]")
    cop_min: float = Field(
        1.5, ge=1.01, le=5.0, description="Physical lower bound on COP [-], must be > 1"
    )
    cop_max: float = Field(
        7.0, ge=2.0, le=15.0, description="Upper bound on COP for fail-fast validation [-]"
    )


_UFH_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "C_r",
    "C_b",
    "R_br",
    "R_ro",
    "eta",
    "internal_gains_heat_fraction",
    "room_temperature_bias_c",
)
_DHW_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "dhw_C_top",
    "dhw_C_bot",
    "dhw_R_strat",
    "dhw_R_loss",
    "dhw_top_temperature_bias_c",
    "dhw_bottom_temperature_bias_c",
    "dhw_boiler_ambient_bias_c",
)
_COP_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "eta_carnot",
    "T_supply_min",
    "T_ref_outdoor_curve",
    "heating_curve_slope",
)


def validate_run_request_physics(req: RunRequest) -> None:
    """Fail fast when a fully materialised runtime request violates coupled physics.

    Args:
        req: Validated runtime request whose scalar fields already satisfy the
            Pydantic field bounds.

    Raises:
        ValueError: If the UFH tuple violates its Forward-Euler validity limits or
            if the DHW tuple/disturbances are non-physical for runtime discretisation.
    """
    # Implements the UFH state-space validity checks from §4/§5.
    ThermalModel(
        ThermalParameters(
            dt_hours=req.dt_hours,
            C_r=req.C_r,
            C_b=req.C_b,
            R_br=req.R_br,
            R_ro=req.R_ro,
            alpha=req.alpha,
            eta=req.eta,
            A_glass=req.A_glass,
        )
    )
    if req.dhw_enabled:
        # DHW runtime uses exact ZOH discretisation; validate only the physical
        # parameter tuple and the supplied tap-flow disturbance.
        dhw_model = DHWModel(
            DHWParameters(
                dt_hours=req.dt_hours,
                C_top=req.dhw_C_top,
                C_bot=req.dhw_C_bot,
                R_strat=req.dhw_R_strat,
                R_loss=req.dhw_R_loss,
                lambda_water=req.dhw_lambda_water_kwh_per_m3k,
            )
        )
        dhw_model.state_matrices(0.0)


def merge_run_request_updates(base_request: RunRequest, updates: dict[str, object]) -> RunRequest:
    """Return a fully revalidated request after applying runtime updates.

    Args:
        base_request: Baseline request whose validated values act as defaults.
        updates: Candidate field updates, for example calibrated parameters or
            live sensor readings.

    Returns:
        A new validated :class:`RunRequest`.

    Raises:
        ValueError: If Pydantic validation or coupled physical runtime checks fail.
    """
    if not updates:
        validate_run_request_physics(base_request)
        return base_request
    merged_request = RunRequest.model_validate({**base_request.model_dump(mode="python"), **updates})
    validate_run_request_physics(merged_request)
    return merged_request


def sanitize_calibration_overrides(
    base_request: RunRequest,
    overrides: CalibrationParameterOverrides,
) -> tuple[CalibrationParameterOverrides, dict[str, str]]:
    """Keep only calibration override groups that remain valid for runtime MPC.

    The calibrated UFH tuple ``(C_r, C_b, R_br, R_ro)`` is treated as one coupled
    physical group. Rejecting or clipping only a single field would destroy the
    fitted RC model, so the whole group is dropped whenever it cannot be applied
    to the runtime request at the configured MPC timestep.

    Args:
        base_request: Baseline runtime request onto which overrides would be applied.
        overrides: Calibrated parameter snapshot read from persistence or produced
            by the automatic calibration scheduler.

    Returns:
        A tuple ``(accepted_overrides, rejection_reasons)`` where
        ``accepted_overrides`` contains only the runtime-safe groups and
        ``rejection_reasons`` maps rejected group names to explicit failure messages.
    """
    accepted_overrides = CalibrationParameterOverrides()
    rejection_reasons: dict[str, str] = {}
    override_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ufh", _UFH_CALIBRATION_OVERRIDE_FIELDS),
        ("dhw", _DHW_CALIBRATION_OVERRIDE_FIELDS),
        ("cop", _COP_CALIBRATION_OVERRIDE_FIELDS),
    )

    current_request = base_request
    raw_updates = overrides.as_run_request_updates()
    for group_name, fields in override_groups:
        group_updates = {field: raw_updates[field] for field in fields if field in raw_updates}
        if not group_updates:
            continue
        try:
            current_request = merge_run_request_updates(current_request, group_updates)
        except Exception as exc:  # noqa: BLE001
            rejection_reasons[group_name] = str(exc)
            continue
        accepted_overrides = accepted_overrides.merged_with(
            CalibrationParameterOverrides.model_validate(group_updates)
        )
    return accepted_overrides, rejection_reasons


def build_safe_calibration_overrides(
    base_request: RunRequest,
    repository: "TelemetryRepository",
) -> dict[str, object]:
    """Return the latest runtime-safe calibration overrides for one request.

    The persisted calibration snapshot may contain parameter groups that are valid
    for offline fitting but unstable or otherwise incompatible with the current MPC
    timestep. This helper reuses :func:`sanitize_calibration_overrides` so every
    runtime entry point applies the same fail-fast acceptance rules.

    Args:
        base_request: Validated runtime request whose timestep and physical limits
            define the admissible override region.
        repository: Telemetry repository exposing the latest persisted calibration
            snapshot.

    Returns:
        Mapping with only the calibration overrides that remain safe for runtime MPC.
    """

    calibration_snapshot = repository.get_latest_calibration_snapshot()
    if calibration_snapshot is None or not calibration_snapshot.has_effective_parameters:
        return {}

    safe_overrides, rejection_reasons = sanitize_calibration_overrides(
        base_request,
        calibration_snapshot.effective_parameters,
    )
    for group_name, reason in rejection_reasons.items():
        log.warning(
            "Ignoring unsafe calibration override group '%s' for runtime MPC input: %s",
            group_name,
            reason,
        )
    return safe_overrides.as_run_request_updates()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MPCStepResult:
    """Numerical result of one MPC solve step (no charts, no web concerns).

    Returned by :meth:`Optimizer.solve` so HTTP and periodic scheduler paths
    can reuse the same core solver without duplicating logic.

    Attributes:
        solution:       Raw convex-MPC solution with full control sequences.
        ufh_forecast:   Forecast horizon used for UFH (contains COP array).
        dhw_forecast:   Forecast horizon used for DHW, or ``None`` when disabled.
        p_ufh_kw:       Clipped UFH thermal power array [kW], length N.
        p_dhw_kw:       Clipped DHW thermal power array [kW], length N (zeros if DHW off).
        cop_ufh_arr:    UFH COP array used in the solve, length N.
        cop_dhw_arr:    DHW COP array used in the solve, length N.
        pv_kw:          PV generation array [kW], length N.
        total_cost_eur: Total electricity cost over horizon [€].
        ufh_energy_kwh: UFH thermal energy [kWh].
        dhw_energy_kwh: DHW thermal energy [kWh].
        start_hour:     UTC hour the solve was triggered (0–23).
    """

    solution: MPCSolution
    ufh_forecast: ForecastHorizon
    dhw_forecast: DHWForecastHorizon | None
    p_ufh_kw: np.ndarray
    p_dhw_kw: np.ndarray
    cop_ufh_arr: np.ndarray
    cop_dhw_arr: np.ndarray
    pv_kw: np.ndarray
    total_cost_eur: float
    ufh_energy_kwh: float
    dhw_energy_kwh: float
    start_hour: int


@dataclass(frozen=True, slots=True)
class ScheduledRunSnapshot:
    """Immutable snapshot of the latest successful scheduled MPC execution.

    Attributes:
        solved_at_utc: UTC timestamp when the scheduled step finished.
        request: Fully materialized optimizer input used for this run.
        result: Numerical optimizer output for this run.
    """

    solved_at_utc: datetime
    request: RunRequest
    result: MPCStepResult


# ---------------------------------------------------------------------------
# Optimizer — the domain core
# ---------------------------------------------------------------------------


class Optimizer:
    """Stateless MPC optimisation service.

    Orchestrates the full solve pipeline:

    1. Build the UFH :class:`~home_optimizer.domain.ufh.model.ThermalModel`.
    2. Build the Carnot :class:`~home_optimizer.domain.heat_pump.cop.HeatPumpCOPModel` (§14.1).
    3. Compute time-varying COP arrays over the MPC horizon.
    4. Construct UFH and DHW :class:`~home_optimizer.types.ForecastHorizon` objects.
    5. Optionally build the :class:`~home_optimizer.domain.dhw.model.DHWModel`.
    6. Solve the QP via :class:`~home_optimizer.control.mpc.MPCController` (CVXPY only).
    7. Compute energy and electricity-cost summaries.

    The class is *stateless* between calls: no mutable attributes are
    modified by :meth:`solve`.  It is therefore safe to share a single
    instance across threads (e.g. from the FastAPI dependency-injection
    container and the APScheduler background thread).

    Example::

        optimizer = Optimizer()
        result = optimizer.solve(req)
        print(result.p_ufh_kw[0])  # first-step UFH power [kW]
    """

    _latest_scheduled_snapshot: ScheduledRunSnapshot | None = None
    _snapshot_lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, req: RunRequest) -> MPCStepResult:
        """Run one full MPC optimisation step.

        Args:
            req: :class:`RunRequest` with all physical and MPC parameters.
                 Callers are responsible for validating the parameters before
                 passing them (e.g. via Pydantic in the API layer).

        Returns:
            :class:`MPCStepResult` with the full solution and numerical
            summaries.  No Plotly charts — visualisation is the API's concern.

        Raises:
            ValueError: If any derived parameter violates a physical constraint
                (e.g. COP ≤ 1 after Carnot pre-calculation, or no convex solver
                reaches an optimal MPC solution).
        """
        start_hour = datetime.now(tz=timezone.utc).hour
        return OptimizerPipeline.solve(req, start_hour=start_hour)

    def run_scheduled_once(
        self,
        base_input: RunRequest,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
    ) -> MPCStepResult | None:
        """Execute one periodic MPC step and log first-step actions.

        This method is intended for APScheduler jobs. It keeps the legacy
        scheduler behavior: sensor-read failures and solver failures are logged
        and skipped for the current interval.

        Args:
            base_input:  Fully populated request containing all physical and MPC
                parameters; live sensors override selected initial conditions.
            backend:     Optional sensor backend providing live readings.
            repository:  Optional telemetry repository.  When provided, the most
                recently persisted Open-Meteo forecast is injected as real
                ``t_out_forecast``, ``gti_window_forecast``, and
                ``gti_pv_forecast`` arrays, replacing the proxy sine-curves.

        Returns:
            :class:`MPCStepResult` on success; ``None`` when the step is
            skipped due to a recoverable read/solve failure.
        """
        try:
            optimizer_input = self._build_scheduled_input(
                base_input=base_input, backend=backend, repository=repository
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("MPC run skipped — sensor read failed: %s", exc)
            return None

        try:
            result = self.solve(optimizer_input)
        except Exception as exc:  # noqa: BLE001
            log.error("MPC solve failed: %s", exc)
            return None

        sol = result.solution
        with type(self)._snapshot_lock:
            type(self)._latest_scheduled_snapshot = ScheduledRunSnapshot(
                solved_at_utc=datetime.now(tz=timezone.utc),
                request=optimizer_input,
                result=result,
            )
        log.info(
            "MPC step complete | status=%s | obj=%.3f | "
            "P_UFH[0]=%.2f kW | P_dhw[0]=%.2f kW | cost=%.4f EUR",
            sol.solver_status,
            sol.objective_value,
            result.p_ufh_kw[0],
            result.p_dhw_kw[0],
            result.total_cost_eur,
        )
        return result

    @classmethod
    def get_latest_scheduled_snapshot(cls) -> ScheduledRunSnapshot | None:
        """Return the most recent successful scheduled MPC snapshot.

        Returns:
            :class:`ScheduledRunSnapshot` when at least one periodic run
            succeeded in this process; otherwise ``None``.
        """
        with cls._snapshot_lock:
            return cls._latest_scheduled_snapshot

    @classmethod
    def clear_latest_scheduled_snapshot(cls) -> None:
        """Clear cached scheduled snapshot (mainly used by tests)."""
        with cls._snapshot_lock:
            cls._latest_scheduled_snapshot = None

    def schedule_periodic(
        self,
        base_input: RunRequest,
        scheduler: "BackgroundScheduler",
        interval_seconds: int,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
        run_immediately: bool = True,
    ) -> None:
        """Register periodic MPC execution on an APScheduler scheduler.

        Args:
            base_input:       Immutable request template for periodic solves.
            scheduler:        Running APScheduler background scheduler.
            interval_seconds: Period between MPC runs [s], must be > 0.
            backend:          Optional sensor backend used for live overrides.
            repository:       Optional telemetry repository used to inject the
                latest Open-Meteo forecast arrays (t_out, GTI window, GTI PV)
                into each scheduled run.  When ``None`` the proxy sine-curves
                are used as fallback.
            run_immediately:  Run one synchronous step before scheduling.

        Raises:
            ValueError: If ``interval_seconds`` is not strictly positive.
        """
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}. "
                "Pass 0 to disable MPC scheduling (do not call this method)."
            )

        if run_immediately:
            log.info("Running initial MPC step before scheduling periodic job...")
            self.run_scheduled_once(base_input=base_input, backend=backend, repository=repository)

        scheduler.add_job(
            self.run_scheduled_once,
            trigger="interval",
            seconds=interval_seconds,
            kwargs={"base_input": base_input, "backend": backend, "repository": repository},
            id="mpc_periodic",
            replace_existing=True,
            misfire_grace_time=max(1, interval_seconds // 2),
        )
        log.info(
            "MPC periodic job scheduled: every %d s (%d min)",
            interval_seconds,
            interval_seconds // 60,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _materialize_horizon_array(
        *,
        name: str,
        horizon_steps: int,
        values: list[float] | None,
        fallback_scalar: float | None = None,
    ) -> np.ndarray:
        """Return a full-horizon forecast array with fail-fast length validation.

        Args:
            name: Human-readable forecast name for error messages.
            horizon_steps: Required horizon length ``N`` [-].
            values: Optional explicit forecast samples. When provided, the array
                must contain at least ``N`` values; only the first ``N`` are used.
            fallback_scalar: Optional scalar value that is broadcast to length ``N``
                when ``values`` is absent. This is reserved for request fields that
                are explicitly scalar by design (e.g. current outdoor temperature).

        Returns:
            NumPy array with shape ``(N,)`` and dtype ``float``.

        Raises:
            ValueError: If the provided forecast is not one-dimensional or does not
                contain enough samples for the MPC horizon.
        """
        return ForecastBuilder.materialize_horizon_array(
            name=name,
            horizon_steps=horizon_steps,
            values=values,
            fallback_scalar=fallback_scalar,
        )

    @staticmethod
    def _build_cop_model(req: RunRequest) -> HeatPumpCOPModel:
        """Construct the Carnot COP model from the user request.

        Assembles a ``HeatPumpCOPParameters`` dataclass from the request fields
        and wraps it in a ``HeatPumpCOPModel``.  Validation (e.g. cop_min > 1,
        eta_carnot ∈ (0,1]) is handled by ``HeatPumpCOPParameters.__post_init__``.

        Args:
            req: Validated Pydantic request containing all COP model parameters.

        Returns:
            Constructed and validated :class:`~home_optimizer.domain.heat_pump.cop.HeatPumpCOPModel`.

        Raises:
            ValueError: If any parameter violates a physical constraint.
        """
        return OptimizerPipeline.build_cop_model(req)

    @staticmethod
    def _build_ufh_forecast(
        req: RunRequest,
        start_hour: int,
        cop_model: HeatPumpCOPModel,
    ) -> ForecastHorizon:
        """Build the UFH disturbance and price forecast over the horizon.

        Injects the time-varying UFH COP array (computed from the Carnot model
        and outdoor temperature) so the MPC can use physical electricity costs.
        The electricity price array is constructed by the configured
        :class:`~home_optimizer.pricing.BasePriceModel` subclass
        (flat / dual-tariff / Nordpool), ensuring no magic numbers appear here.

        Args:
            req:        Validated request with all forecast parameters.
            start_hour: Current hour of the day (0–23), used to index the price
                        pattern and solar profile.
            cop_model:  Carnot COP model; used to compute ``cop_ufh_k`` array.

        Returns:
            :class:`~home_optimizer.types.ForecastHorizon` with N steps,
            including the COP array.
        """
        return ForecastBuilder.build_ufh_forecast(
            req,
            start_hour=start_hour,
            cop_model=cop_model,
        )

    @staticmethod
    def _build_internal_gains_profile(req: RunRequest, *, horizon_steps: int) -> np.ndarray:
        """Return the internal-gains horizon using explicit forecast or physical baseload mapping.

        Precedence:
        1. explicit ``internal_gains_forecast``
        2. ``max(internal_gains_kw, internal_gains_heat_fraction × baseload_forecast)``
        3. scalar ``internal_gains_kw`` broadcast across the horizon

        The baseload mapping expresses that only a fraction of electrical household
        demand becomes useful sensible heat in the conditioned zone, while the
        scalar baseline captures occupant/metabolic and other non-forecast gains.
        To avoid a second free tuning knob, the electrical baseload reference is
        derived implicitly from ``internal_gains_kw`` and
        ``internal_gains_heat_fraction`` instead of being exposed as a
        separate user parameter.
        """

        return ForecastBuilder.build_internal_gains_profile(
            req,
            horizon_steps=horizon_steps,
        )

    @staticmethod
    def _map_baseload_to_internal_gains_increment(
        *,
        baseload_profile_kw: np.ndarray,
        baseline_internal_gains_kw: float,
        heat_fraction: float,
    ) -> np.ndarray:
        """Return the incremental sensible heat gain implied by the electrical baseload.

        Implements the optimizer-side mapping

        ``Q_int[k] = max(Q_int,baseline, heat_fraction * P_baseload[k])``

        from the project-specific grey-box interpretation of internal gains.  The
        implicit electrical baseload reference is derived as

        ``P_reference = Q_int,baseline / heat_fraction``

        for ``heat_fraction > 0``. This keeps only one tuned baseline term in the
        request model and avoids separate calibration of two overlapping background
        offsets. The returned increment therefore equals

        ``max(heat_fraction * P_baseload[k] - Q_int,baseline, 0)``.

        Args:
            baseload_profile_kw: Forecast non-heat-pump household electrical demand
                over the horizon [kW], shape ``(N,)``.
            baseline_internal_gains_kw: Scalar background internal-gains level
                already present in the UFH disturbance model [kW].
            heat_fraction: Fraction of the excess electrical demand that appears
                indoors as useful sensible heat [-].

        Returns:
            Incremental internal-gains profile [kW], shape ``(N,)``.

        Raises:
            ValueError: If the baseload profile contains negative values or when
                ``heat_fraction`` is negative.
        """

        return ForecastBuilder.map_baseload_to_internal_gains_increment(
            baseload_profile_kw=baseload_profile_kw,
            baseline_internal_gains_kw=baseline_internal_gains_kw,
            heat_fraction=heat_fraction,
        )

    @staticmethod
    def _build_dhw_forecast(
        req: RunRequest,
        N: int,
        cop_model: HeatPumpCOPModel,
    ) -> DHWForecastHorizon:
        """Build the DHW disturbance forecast over the horizon.

        The DHW COP depends on the outdoor temperature (evaporator side) and the
        hot-water supply temperature (condenser side ≈ T_dhw_min for normal
        operation).  Time-varying COP is injected into the forecast.

        Args:
            req:       Validated request with all DHW parameters.
            N:         Horizon length [steps].
            cop_model: Carnot COP model; used to compute ``cop_dhw_k`` array.

        Returns:
            :class:`~home_optimizer.types.DHWForecastHorizon` with N steps,
            including the COP array.
        """
        return ForecastBuilder.build_dhw_forecast(
            req,
            horizon_steps=N,
            cop_model=cop_model,
        )

    @staticmethod
    def _build_scheduled_input(
        base_input: RunRequest,
        backend: "SensorBackend | None",
        repository: "TelemetryRepository | None" = None,
    ) -> RunRequest:
        """Build one solver input by applying optional live sensor and forecast overrides.

        Overrides applied (in order):

        1. **Persisted calibration snapshot** (via ``repository``): latest
           calibrated physical/COP parameters stored by the addon.
        2. **Live sensor readings** (via ``backend``): room temperature,
           slab temperature estimate, outdoor temperature, DHW temperatures.
        3. **Open-Meteo forecast arrays** (via ``repository``): fetches the most
           recently persisted :class:`~home_optimizer.telemetry.models.ForecastSnapshot`
           batch and injects ``t_out_forecast``, ``gti_window_forecast``, and
           ``gti_pv_forecast`` so the MPC uses real forecast data instead of
           the proxy sine-curves.

        Args:
            base_input:  Baseline validated request for periodic execution.
            backend:     Optional sensor backend. When ``None``, initial-state
                overrides are skipped.
            repository:  Optional telemetry repository. When ``None`` or when
                the forecast table is empty, forecast arrays remain ``None``
                and the proxy fallback in ``_build_ufh_forecast`` is used.

        Returns:
            Immutable :class:`RunRequest` for one optimizer step.

        Raises:
            Exception: Re-raised backend read failures for fail-fast scheduling.
        """
        overrides: dict = {}

        # ── 1. Persisted calibration overrides ────────────────────────────
        if repository is not None:
            try:
                overrides.update(build_safe_calibration_overrides(base_input, repository))
            except Exception as exc:  # noqa: BLE001
                log.warning("Calibration snapshot DB read failed: %s", exc)

        # ── 2. Live sensor overrides ─────────────────────────────────────
        if backend is not None:
            effective_request = merge_run_request_updates(base_input, overrides)
            readings = backend.read_all()
            corrected_room_temperature_c = (
                readings.room_temperature_c + effective_request.room_temperature_bias_c
            )

            # Estimate slab temperature from supply/return average when possible.
            t_b_estimate = (
                (readings.hp_supply_temperature_c + readings.hp_return_temperature_c) / 2.0
                if readings.hp_supply_temperature_c > corrected_room_temperature_c
                else corrected_room_temperature_c + 2.0
            )

            overrides.update(
                {
                    "T_r_init": corrected_room_temperature_c,
                    "T_b_init": t_b_estimate,
                    "outdoor_temperature_c": readings.outdoor_temperature_c,
                    "shutter_living_room_pct": readings.shutter_living_room_pct,
                }
            )

            if base_input.dhw_enabled:
                overrides["dhw_T_top_init"] = (
                    readings.dhw_top_temperature_c + effective_request.dhw_top_temperature_bias_c
                )
                overrides["dhw_T_bot_init"] = (
                    readings.dhw_bottom_temperature_c + effective_request.dhw_bottom_temperature_bias_c
                )
                overrides["dhw_t_mains_c"] = readings.t_mains_estimated_c
                overrides["dhw_t_amb_c"] = (
                    readings.boiler_ambient_temp_c + effective_request.dhw_boiler_ambient_bias_c
                )

        # ── 3. Open-Meteo forecast arrays from database ──────────────────
        if repository is not None:
            try:
                rows, forecast_overrides = build_repository_forecast_overrides(
                    base_input,
                    repository,
                    existing=overrides,
                )
                if rows:
                    overrides.update(forecast_overrides)
                    log.debug(
                        "Injected Open-Meteo forecast (%d steps) into MPC run.", len(rows[:base_input.horizon_hours])
                    )
                else:
                    log.debug("No forecast rows in DB — real GTI data unavailable.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Forecast DB read failed: %s", exc)

        if not overrides:
            return merge_run_request_updates(base_input, {})
        return merge_run_request_updates(base_input, overrides)
