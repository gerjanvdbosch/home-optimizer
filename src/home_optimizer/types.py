"""Validated parameter dataclasses and forecast containers.

Units
-----
Temperatures         : °C
Thermal capacity     : kWh/K
Thermal resistance   : K/kW
Power                : kW
Irradiance           : W/m²  (converted to kW via W_PER_KW inside ForecastHorizon)
Electricity price    : €/kWh
Flow rate            : m³/h
"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Array1DInput = Iterable[float] | np.ndarray

# ---------------------------------------------------------------------------
# Universal unit-conversion constants (named, never hardcoded inline)
# ---------------------------------------------------------------------------

#: 1 kW = 1000 W  — used to convert W/m² irradiance to kW.
W_PER_KW: float = 1000.0

#: Absolute zero expressed in degrees Celsius. Temperatures below this are non-physical.
ABSOLUTE_ZERO_C: float = -273.15

#: Water volumetric heat capacity λ = ρ·c_p [kWh/(m³·K)] (§8.4, §15).
LAMBDA_WATER_KWH_PER_M3_K: float = 1.1628

#: Number of litres in one cubic metre [L/m³].
LITERS_PER_CUBIC_METER: float = 1000.0

#: Cubic metres in one litre [m³/L].
M3_PER_LITER: float = 1.0 / LITERS_PER_CUBIC_METER


class CalibrationParameterOverrides(BaseModel):
    """Validated calibrated parameter overrides that can be applied to ``RunRequest``.

    The automatic calibration pipeline stores only parameters that are directly
    usable by the runtime MPC/COP models. Every field is optional so a stage can
    update only the parameters it actually identified while previous successful
    values remain active.
    """

    model_config = ConfigDict(extra="forbid")

    C_r: float | None = Field(default=None, gt=0.0, description="UFH room capacity C_r [kWh/K]")
    C_b: float | None = Field(default=None, gt=0.0, description="UFH slab capacity C_b [kWh/K]")
    R_br: float | None = Field(default=None, gt=0.0, description="UFH floor-room resistance R_br [K/kW]")
    R_ro: float | None = Field(default=None, gt=0.0, description="UFH room-outdoor resistance R_ro [K/kW]")
    dhw_R_strat: float | None = Field(
        default=None,
        gt=0.0,
        description="DHW stratification resistance R_strat [K/kW]",
    )
    dhw_R_loss: float | None = Field(
        default=None,
        gt=0.0,
        description="DHW standby-loss resistance R_loss [K/kW]",
    )
    eta_carnot: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Shared Carnot efficiency eta_carnot [-]",
    )
    T_supply_min: float | None = Field(
        default=None,
        description="UFH minimum supply temperature T_supply_min [°C]",
    )
    heating_curve_slope: float | None = Field(
        default=None,
        ge=0.0,
        description="UFH heating-curve slope [K/K]",
    )

    def as_run_request_updates(self) -> dict[str, float]:
        """Return only the non-null fields as ``RunRequest.model_copy`` updates."""
        return self.model_dump(exclude_none=True)

    def merged_with(self, newer: "CalibrationParameterOverrides") -> "CalibrationParameterOverrides":
        """Return ``self`` with any non-null values from ``newer`` applied."""
        return type(self).model_validate(
            {
                **self.model_dump(),
                **newer.model_dump(exclude_none=True),
            }
        )


class CalibrationStageResult(BaseModel):
    """Summary of one automatic calibration stage.

    The payload is intentionally compact so it can be stored in SQLite JSON and
    exposed verbatim through the API for observability.
    """

    model_config = ConfigDict(extra="forbid")

    stage_name: str = Field(min_length=1)
    succeeded: bool
    message: str = Field(min_length=1)
    sample_count: int | None = Field(default=None, ge=0)
    segment_count: int | None = Field(default=None, ge=0)
    dataset_start_utc: datetime | None = None
    dataset_end_utc: datetime | None = None
    optimizer_status: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    overrides: CalibrationParameterOverrides = Field(default_factory=CalibrationParameterOverrides)


class CalibrationSnapshotPayload(BaseModel):
    """Persisted automatic-calibration snapshot consumed by the scheduled MPC path."""

    model_config = ConfigDict(extra="forbid")

    generated_at_utc: datetime
    effective_parameters: CalibrationParameterOverrides = Field(default_factory=CalibrationParameterOverrides)
    ufh_active: CalibrationStageResult | None = None
    dhw_standby: CalibrationStageResult | None = None
    dhw_active: CalibrationStageResult | None = None
    cop: CalibrationStageResult | None = None

    @property
    def has_effective_parameters(self) -> bool:
        """Return ``True`` when at least one calibrated override is available."""
        return bool(self.effective_parameters.as_run_request_updates())


def _as_1d(name: str, values: Array1DInput) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array-like.")
    return arr.copy()


# ---------------------------------------------------------------------------
# Physical / house parameters — UFH
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermalParameters:
    """Physical parameters of the house and underfloor heating system.

    Parameters
    ----------
    dt_hours:   Forward-Euler time step [h].
    C_r:        Room air + furniture thermal capacity [kWh/K].
    C_b:        Floor / concrete slab thermal capacity [kWh/K].
    R_br:       Thermal resistance between floor and room air [K/kW].
    R_ro:       Thermal resistance between room and outside [K/kW].
    alpha:      Fraction of solar gain that heats the room air directly (0–1).
    eta:        Window glass solar transmittance (0–1).
    A_glass:    South-facing glass area [m²].
    """

    dt_hours: float
    C_r: float
    C_b: float
    R_br: float
    R_ro: float
    alpha: float
    eta: float
    A_glass: float

    def __post_init__(self) -> None:
        for f in ("dt_hours", "C_r", "C_b", "R_br", "R_ro", "A_glass"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"{f} must be strictly positive.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1].")

    @property
    def euler_time_constants_hours(self) -> tuple[float, float, float]:
        """Dominant time constants for the Euler stability criterion [h]."""
        return (self.C_r * self.R_br, self.C_b * self.R_br, self.C_r * self.R_ro)

    def max_stable_euler_dt(self, safety_factor: float = 0.2) -> float:
        """Upper bound on dt for a stable forward-Euler step [h]."""
        return safety_factor * min(self.euler_time_constants_hours)

    def assert_euler_stable(self, safety_factor: float = 0.2) -> None:
        """Raise if the current dt_hours exceeds the Euler stability bound."""
        limit = self.max_stable_euler_dt(safety_factor)
        if self.dt_hours > limit:
            raise ValueError(
                f"Forward-Euler time step dt={self.dt_hours:.3f} h exceeds the stability "
                f"bound {limit:.3f} h.  Reduce dt or switch to ZOH discretisation."
            )


# ---------------------------------------------------------------------------
# Physical / tank parameters — DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DHWParameters:
    """Physical parameters of the DHW 2-node stratification tank.

    The tank is split into a top layer (tap-water outlet) and a bottom layer
    (heat-pump inlet).  The heat-pump heat exchanger sits at the bottom (A5).

    Parameters
    ----------
    dt_hours:       Forward-Euler time step [h].
    C_top:          Thermal capacity of the top layer [kWh/K].
    C_bot:          Thermal capacity of the bottom layer [kWh/K].
    R_strat:        Stratification thermal resistance top↔bottom [K/kW].
    R_loss:         Standby-loss resistance to ambient (both layers share R_loss) [K/kW].
    lambda_water:   ρ·c_p of water [kWh/(m³·K)] (§8.4).
    """

    dt_hours: float
    C_top: float
    C_bot: float
    R_strat: float
    R_loss: float
    lambda_water: float = LAMBDA_WATER_KWH_PER_M3_K

    def __post_init__(self) -> None:
        for f in ("dt_hours", "C_top", "C_bot", "R_strat", "R_loss", "lambda_water"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"{f} must be strictly positive.")

    @property
    def euler_time_constants_hours(self) -> tuple[float, float, float]:
        """Dominant time constants for the Euler stability criterion [h]."""
        return (self.C_top * self.R_strat, self.C_bot * self.R_strat, self.C_top * self.R_loss)

    def max_stable_euler_dt(self, safety_factor: float = 0.2) -> float:
        """Upper bound on dt for a stable forward-Euler step [h]."""
        if safety_factor <= 0.0:
            raise ValueError("safety_factor must be strictly positive.")
        return safety_factor * min(self.euler_time_constants_hours)

    def assert_euler_stable(self, safety_factor: float = 0.2) -> None:
        """Raise if the current dt_hours exceeds the Euler stability bound."""
        limit = self.max_stable_euler_dt(safety_factor)
        if self.dt_hours > limit:
            raise ValueError(
                f"Forward-Euler time step dt={self.dt_hours:.3f} h exceeds the stability "
                f"bound {limit:.3f} h.  Reduce dt or switch to ZOH discretisation."
            )

    def tap_flow_time_constant_hours(self, v_tap_m3_per_h: float) -> float:
        """Return the DHW top-layer tap time constant ``C_top / (λ·V_tap)`` [h].

        This is the additional DHW-specific stability limit from §10.2: large tap
        flow can create dynamics much faster than the standby/stratification time
        constants, so the discretisation must also remain small relative to this
        tap-driven time constant.
        """
        if v_tap_m3_per_h < 0.0:
            raise ValueError("v_tap_m3_per_h must be non-negative.")
        if v_tap_m3_per_h == 0.0:
            return float("inf")
        return self.C_top / (self.lambda_water * v_tap_m3_per_h)

    def max_stable_euler_dt_for_tap_flow(
        self,
        v_tap_m3_per_h: float,
        safety_factor: float = 0.2,
    ) -> float:
        """Return the most restrictive Euler bound [h] for the supplied tap flow.

        The bound is the minimum of the linear thermal time constants and the
        tap-flow-driven time constant from §10.2.
        """
        base_limit = self.max_stable_euler_dt(safety_factor)
        tap_tau_hours = self.tap_flow_time_constant_hours(v_tap_m3_per_h)
        if np.isinf(tap_tau_hours):
            return base_limit
        return min(base_limit, safety_factor * tap_tau_hours)

    def assert_euler_stable_for_tap_flow(
        self,
        v_tap_m3_per_h: float,
        safety_factor: float = 0.2,
    ) -> None:
        """Raise when ``dt_hours`` violates the DHW Euler bound at a given tap flow.

        This check strengthens :meth:`assert_euler_stable` by including the
        tap-flow-dependent limit ``Δt << C_top / (λ·V_tap,max)`` from §10.2.
        """
        limit = self.max_stable_euler_dt_for_tap_flow(v_tap_m3_per_h, safety_factor)
        if self.dt_hours > limit:
            raise ValueError(
                f"Forward-Euler time step dt={self.dt_hours:.3f} h exceeds the DHW stability "
                f"bound {limit:.3f} h at V_tap={v_tap_m3_per_h:.6f} m³/h. "
                "Reduce dt, reduce the admissible tap flow, or switch to ZOH discretisation."
            )


# ---------------------------------------------------------------------------
# MPC parameters — UFH
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MPCParameters:
    """Settings for the UFH Model Predictive Controller.

    Parameters
    ----------
    horizon_steps:   Number of look-ahead steps N.
    Q_c:             Comfort weighting on (T_r - T_ref)² [dimensionless].
    R_c:             Regularisation weighting on P_UFH² (prevents power spikes).
    Q_N:             Terminal comfort weighting.
    P_max:           Maximum UFH **thermal** power [kW].
    delta_P_max:     Maximum ramp-rate per step [kW/step].
    T_min:           Minimum allowed room temperature [°C].
    T_max:           Maximum allowed room temperature [°C].
    cop_ufh:         Coefficient of Performance, UFH mode [dimensionless].
                     The electrical power drawn is P_UFH / cop_ufh.
                     Must satisfy 1 < cop_ufh ≤ cop_max (Fail-Fast, §14.1).
    cop_max:         Upper bound on COP for Fail-Fast validation [dimensionless].
                     A COP above this value indicates a sensor or model error.
    rho_factor:      Soft-constraint penalty multiplier: ρ = rho_factor × max(Q_c, 1).
    """

    horizon_steps: int
    Q_c: float
    R_c: float
    Q_N: float
    P_max: float
    delta_P_max: float
    T_min: float
    T_max: float
    cop_ufh: float
    cop_max: float
    rho_factor: float = 1000.0

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be ≥ 1.")
        for f in ("Q_c", "R_c", "Q_N"):
            if getattr(self, f) < 0.0:
                raise ValueError(f"{f} must be non-negative.")
        for f in ("P_max", "delta_P_max"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"{f} must be strictly positive.")
        if self.T_min > self.T_max:
            raise ValueError("T_min must be ≤ T_max.")
        if self.rho_factor <= 0.0:
            raise ValueError("rho_factor must be strictly positive.")
        # §14.1 Fail-Fast COP validation: COP must be physically meaningful
        if self.cop_max <= 1.0:
            raise ValueError("cop_max must be strictly greater than 1.")
        if self.cop_ufh <= 1.0:
            raise ValueError(f"cop_ufh={self.cop_ufh} is physically impossible: COP must be > 1.")
        if self.cop_ufh > self.cop_max:
            raise ValueError(f"cop_ufh={self.cop_ufh} exceeds cop_max={self.cop_max}.")


# ---------------------------------------------------------------------------
# MPC parameters — DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DHWMPCParameters:
    """MPC settings for the DHW subsystem.

    Parameters
    ----------
    P_dhw_max:              Maximum **thermal** power to the bottom layer [kW].
    delta_P_dhw_max:        Maximum ramp-rate per step [kW/step].
    T_dhw_min:              Minimum tap (top-layer) temperature for comfort [°C].
    T_legionella:           Legionella prevention temperature [°C] (typically 60).
    legionella_period_steps:  Steps between mandatory legionella cycles.
    legionella_duration_steps: Minimum consecutive steps at T_legionella.
    cop_dhw:                Coefficient of Performance, DHW mode [dimensionless].
                            The electrical power drawn is P_dhw / cop_dhw.
                            Must satisfy 1 < cop_dhw ≤ cop_max (Fail-Fast, §14.1).
    cop_max:                Upper bound on COP for Fail-Fast validation [dimensionless].
    comfort_rho_factor:     Soft-constraint penalty multiplier for T_top < T_dhw_min.
    legionella_rho_factor:  Soft-constraint penalty multiplier for T_top < T_legionella.
    """

    P_dhw_max: float
    delta_P_dhw_max: float
    T_dhw_min: float
    T_legionella: float
    legionella_period_steps: int
    legionella_duration_steps: int
    cop_dhw: float
    cop_max: float
    comfort_rho_factor: float = 1000.0
    legionella_rho_factor: float = 1e6

    def __post_init__(self) -> None:
        for f in ("P_dhw_max", "delta_P_dhw_max"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"{f} must be strictly positive.")
        if self.T_dhw_min <= 0.0:
            raise ValueError("T_dhw_min must be positive.")
        if self.T_legionella <= self.T_dhw_min:
            raise ValueError("T_legionella must be strictly greater than T_dhw_min.")
        if self.legionella_period_steps <= 0:
            raise ValueError("legionella_period_steps must be >= 1.")
        if self.legionella_duration_steps <= 0:
            raise ValueError("legionella_duration_steps must be >= 1.")
        if self.comfort_rho_factor <= 0.0:
            raise ValueError("comfort_rho_factor must be strictly positive.")
        if self.legionella_rho_factor <= 0.0:
            raise ValueError("legionella_rho_factor must be strictly positive.")
        # §14.1 Fail-Fast COP validation
        if self.cop_max <= 1.0:
            raise ValueError("cop_max must be strictly greater than 1.")
        if self.cop_dhw <= 1.0:
            raise ValueError(f"cop_dhw={self.cop_dhw} is physically impossible: COP must be > 1.")
        if self.cop_dhw > self.cop_max:
            raise ValueError(f"cop_dhw={self.cop_dhw} exceeds cop_max={self.cop_max}.")


# ---------------------------------------------------------------------------
# MPC parameters — Combined UFH + DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CombinedMPCParameters:
    """Parameters for the combined UFH + DHW Model Predictive Controller.

    Parameters
    ----------
    ufh:              UFH-specific MPC parameters (horizon, weights, bounds, cop_ufh).
    dhw:              DHW-specific MPC parameters (bounds, legionella, cop_dhw).
    P_hp_max_elec:    Maximum total heat-pump **electrical** power budget [kW].
                      The constraint enforced is:
                          P_UFH / COP_UFH + P_dhw / COP_dhw ≤ P_hp_max_elec
                      This correctly models a shared heat pump limited by its
                      electrical capacity (§14, shared WP constraint).
    """

    ufh: MPCParameters
    dhw: DHWMPCParameters
    P_hp_max_elec: float

    def __post_init__(self) -> None:
        if self.P_hp_max_elec <= 0.0:
            raise ValueError("P_hp_max_elec must be strictly positive.")


# ---------------------------------------------------------------------------
# Kalman filter noise parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KalmanNoiseParameters:
    """Noise covariance parameters for a 2-state Kalman filter.

    Parameters
    ----------
    process_covariance:    2×2 process noise covariance Q_n [K²].
                           May also be given as a length-2 diagonal vector.
    measurement_variance:  Scalar measurement noise variance R_n [K²].
    """

    process_covariance: np.ndarray
    measurement_variance: float

    def __post_init__(self) -> None:
        q = np.asarray(self.process_covariance, dtype=float)
        if q.shape == (2,):
            q = np.diag(q)
        if q.shape != (2, 2):
            raise ValueError("process_covariance must be 2×2 or a length-2 diagonal.")
        if not np.allclose(q, q.T):
            raise ValueError("process_covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(q)) < -1e-10:
            raise ValueError("process_covariance must be positive semi-definite.")
        if self.measurement_variance <= 0.0:
            raise ValueError("measurement_variance must be strictly positive.")
        object.__setattr__(self, "process_covariance", q)


# ---------------------------------------------------------------------------
# EKF noise parameters — DHW augmented state (§12 of spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EKFNoiseParameters:
    """Noise covariance parameters for the DHW Extended Kalman Filter.

    The augmented state is x_aug = [T_top, T_bot, V_tap]ᵀ.  The combined
    3×3 process-noise covariance is block-diagonal:

        Q_aug = diag(Q_n_dhw, Q_n_Vtap)   ∈ ℝ³ˣ³

    where Q_n_dhw is a 2×2 matrix for the temperature states and Q_n_Vtap
    is a scalar for the flow-rate state (random-walk model, §12.2).

    Both temperature sensors are available, so R_n_dhw is a 2×2 diagonal
    matrix.

    Parameters
    ----------
    process_cov_temperatures:
        2×2 process noise covariance Q_n_dhw for [T_top, T_bot] [K²].
        May also be given as a length-2 diagonal vector.
    process_var_vtap:
        Scalar process noise variance Q_n_Vtap for V_tap [(m³/h)²].
        Controls how quickly the EKF tracks tap events (higher → faster,
        noisier; lower → slower, smoother).  Must be strictly positive.
    measurement_var_t_top:
        Scalar measurement noise variance σ²_T_top [K²].
    measurement_var_t_bot:
        Scalar measurement noise variance σ²_T_bot [K²].
    """

    process_cov_temperatures: np.ndarray
    process_var_vtap: float
    measurement_var_t_top: float
    measurement_var_t_bot: float

    def __post_init__(self) -> None:
        q = np.asarray(self.process_cov_temperatures, dtype=float)
        if q.shape == (2,):
            q = np.diag(q)
        if q.shape != (2, 2):
            raise ValueError("process_cov_temperatures must be 2×2 or a length-2 diagonal.")
        if not np.allclose(q, q.T):
            raise ValueError("process_cov_temperatures must be symmetric.")
        if np.min(np.linalg.eigvalsh(q)) < -1e-10:
            raise ValueError("process_cov_temperatures must be positive semi-definite.")
        if self.process_var_vtap <= 0.0:
            raise ValueError("process_var_vtap must be strictly positive.")
        if self.measurement_var_t_top <= 0.0:
            raise ValueError("measurement_var_t_top must be strictly positive.")
        if self.measurement_var_t_bot <= 0.0:
            raise ValueError("measurement_var_t_bot must be strictly positive.")
        object.__setattr__(self, "process_cov_temperatures", q)

    @property
    def Q_aug(self) -> np.ndarray:
        """3×3 block-diagonal process noise covariance Q_aug = diag(Q_n_dhw, Q_n_Vtap)."""
        q = np.zeros((3, 3), dtype=float)
        q[:2, :2] = self.process_cov_temperatures
        q[2, 2] = self.process_var_vtap
        return q

    @property
    def R_n(self) -> np.ndarray:
        """2×2 diagonal measurement noise covariance diag(σ²_T_top, σ²_T_bot) [K²]."""
        return np.diag(
            np.array([self.measurement_var_t_top, self.measurement_var_t_bot], dtype=float)
        )


# ---------------------------------------------------------------------------
# Forecast horizon — UFH
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    """Disturbance forecast and reference trajectory over N steps.

    All arrays must have length N (horizon_steps), except
    room_temperature_ref_c which must have length N+1 to include the terminal
    MPC reference.

    Parameters
    ----------
    outdoor_temperature_c:   T_out forecast [°C],  length N.
    gti_w_per_m2:            Global Tilted Irradiance [W/m²], length N.
    internal_gains_kw:       Q_int forecast [kW], length N.
    price_eur_per_kwh:       Dynamic electricity tariff p[k] [€/kWh], length N.
    room_temperature_ref_c:  Comfort setpoint T_ref [°C], length N+1.
    pv_kw:                   PV generation forecast [kW], length N.
                             ``None`` means no PV installed → zeros.
    cop_ufh_k:               Time-varying COP for UFH mode [dimensionless], length N.
                             If ``None``, the scalar ``MPCParameters.cop_ufh`` is used.
                             Typically a function of T_out[k] (colder outside → lower COP).
                             All values must be > 1 (validated in MPC controller against
                             the cop_max from MPCParameters).
    shutter_pct:             Living-room shutter position [%] over the horizon, length N.
                             100 = fully open, 0 = fully closed.  When provided, the
                             effective solar transmittance at step k becomes:
                                 η_eff[k] = η × (shutter_pct[k] / 100)
                             so Q_solar[k] = A_glass × GTI[k] × η_eff[k] / W_PER_KW (§4).
                             ``None`` means shutters are assumed fully open → η_eff = η.
                             Values must be in [0, 100].
    """

    outdoor_temperature_c: np.ndarray
    gti_w_per_m2: np.ndarray
    internal_gains_kw: np.ndarray
    price_eur_per_kwh: np.ndarray
    room_temperature_ref_c: np.ndarray
    #: PV generation forecast [kW], length N.  ``None`` means no PV installed → zeros.
    pv_kw: np.ndarray | None = None
    #: Time-varying UFH COP over horizon [dimensionless], length N.  ``None`` → use scalar.
    cop_ufh_k: np.ndarray | None = None
    #: Shutter position forecast [%], length N.  ``None`` → fully open (η_eff = η).
    shutter_pct: np.ndarray | None = None

    def __post_init__(self) -> None:
        t_out = _as_1d("outdoor_temperature_c", self.outdoor_temperature_c)
        gti = _as_1d("gti_w_per_m2", self.gti_w_per_m2)
        q_int = _as_1d("internal_gains_kw", self.internal_gains_kw)
        price = _as_1d("price_eur_per_kwh", self.price_eur_per_kwh)
        t_ref = _as_1d("room_temperature_ref_c", self.room_temperature_ref_c)

        n = t_out.size
        for name, arr in (
            ("gti_w_per_m2", gti),
            ("internal_gains_kw", q_int),
            ("price_eur_per_kwh", price),
        ):
            if arr.size != n:
                raise ValueError(f"{name} must have length {n}.")
        if t_ref.size != n + 1:
            raise ValueError(
                "room_temperature_ref_c must have length N+1 (includes terminal reference)."
            )
        if np.any(gti < 0.0):
            raise ValueError("gti_w_per_m2 cannot be negative.")
        if np.any(price < 0.0):
            raise ValueError("price_eur_per_kwh cannot be negative.")

        # PV forecast: None → no PV installed (physically valid: zero generation)
        if self.pv_kw is None:
            pv = np.zeros(n)
        else:
            pv = _as_1d("pv_kw", self.pv_kw)
            if pv.size != n:
                raise ValueError(f"pv_kw must have length {n}.")
            if np.any(pv < 0.0):
                raise ValueError("pv_kw cannot be negative.")

        # Time-varying COP: validate length and physical lower bound (> 1).
        # Upper-bound check against cop_max is deferred to the MPC controller,
        # which has access to MPCParameters.
        if self.cop_ufh_k is not None:
            cop_arr = _as_1d("cop_ufh_k", self.cop_ufh_k)
            if cop_arr.size != n:
                raise ValueError(f"cop_ufh_k must have length {n}.")
            if np.any(cop_arr <= 1.0):
                raise ValueError("cop_ufh_k: all COP values must be > 1 (physical lower bound).")
            object.__setattr__(self, "cop_ufh_k", cop_arr)

        # Shutter forecast: None → fully open (shutter_fraction = 1.0 everywhere).
        if self.shutter_pct is not None:
            shutter = _as_1d("shutter_pct", self.shutter_pct)
            if shutter.size != n:
                raise ValueError(f"shutter_pct must have length {n}.")
            if np.any(shutter < 0.0) or np.any(shutter > 100.0):
                raise ValueError("shutter_pct values must be in [0, 100].")
            object.__setattr__(self, "shutter_pct", shutter)

        object.__setattr__(self, "outdoor_temperature_c", t_out)
        object.__setattr__(self, "gti_w_per_m2", gti)
        object.__setattr__(self, "internal_gains_kw", q_int)
        object.__setattr__(self, "price_eur_per_kwh", price)
        object.__setattr__(self, "room_temperature_ref_c", t_ref)
        object.__setattr__(self, "pv_kw", pv)

    @property
    def horizon_steps(self) -> int:
        return int(self.outdoor_temperature_c.size)

    def solar_gains_kw(self, p: ThermalParameters) -> np.ndarray:
        """Convert GTI forecast to solar gain [kW], accounting for shutter position.

        Without shutter data (``shutter_pct is None``):
            Q_solar = A_glass × GTI × η / W_PER_KW

        With shutter data (§4):
            η_eff[k] = η × (shutter_pct[k] / 100)
            Q_solar[k] = A_glass × GTI[k] × η_eff[k] / W_PER_KW

        Args:
            p: Thermal parameters supplying A_glass [m²], η [-], and
               the W_PER_KW conversion constant.

        Returns:
            Solar gain array [kW], shape (N,).  Non-negative by construction
            because GTI ≥ 0 and η_eff ∈ [0, 1].
        """
        if self.shutter_pct is None:
            # Shutters fully open — standard formula.
            eta_eff = p.eta
        else:
            # Dynamic effective transmittance: η_eff[k] = η × fraction[k].
            eta_eff = p.eta * (self.shutter_pct / 100.0)
        return p.A_glass * self.gti_w_per_m2 * eta_eff / W_PER_KW

    def disturbance_matrix(self, p: ThermalParameters) -> np.ndarray:
        """Return N×3 matrix with columns [T_out, Q_solar, Q_int]."""
        return np.column_stack(
            [
                self.outdoor_temperature_c,
                self.solar_gains_kw(p),
                self.internal_gains_kw,
            ]
        )

    @classmethod
    def constant(
        cls,
        horizon_steps: int,
        outdoor_temperature_c: float = 10.0,
        gti_w_per_m2: float = 0.0,
        internal_gains_kw: float = 0.3,
        price_eur_per_kwh: float = 0.25,
        room_temperature_ref_c: float = 21.0,
        pv_kw: float = 0.0,
    ) -> "ForecastHorizon":
        """Convenience factory: constant disturbances over the full horizon."""
        n = horizon_steps
        return cls(
            outdoor_temperature_c=np.full(n, outdoor_temperature_c),
            gti_w_per_m2=np.full(n, gti_w_per_m2),
            internal_gains_kw=np.full(n, internal_gains_kw),
            price_eur_per_kwh=np.full(n, price_eur_per_kwh),
            room_temperature_ref_c=np.full(n + 1, room_temperature_ref_c),
            pv_kw=np.full(n, pv_kw),
        )


# ---------------------------------------------------------------------------
# Forecast horizon — DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DHWForecastHorizon:
    """DHW disturbance forecast over N steps.

    Parameters
    ----------
    v_tap_m3_per_h:      Tap-water flow rate V_tap [m³/h], length N. Non-negative.
    t_mains_c:           Cold mains-water temperature T_mains [°C], length N.
    t_amb_c:             Ambient temperature around the boiler T_amb [°C], length N.
    legionella_required: Boolean mask — True if T_top ≥ T_legionella is required,
                         length N.  Set by an external legionella scheduler.
    cop_dhw_k:           Time-varying COP for DHW mode [dimensionless], length N.
                         If ``None``, the scalar ``DHWMPCParameters.cop_dhw`` is used.
                         Typically a function of T_bot[k] (higher target temp → lower COP).
                         All values must be > 1 (validated in MPC controller against
                         the cop_max from DHWMPCParameters).
    """

    v_tap_m3_per_h: np.ndarray
    t_mains_c: np.ndarray
    t_amb_c: np.ndarray
    legionella_required: np.ndarray
    #: Time-varying DHW COP over horizon [dimensionless], length N.  ``None`` → use scalar.
    cop_dhw_k: np.ndarray | None = None

    def __post_init__(self) -> None:
        v_tap = _as_1d("v_tap_m3_per_h", self.v_tap_m3_per_h)
        t_mains = _as_1d("t_mains_c", self.t_mains_c)
        t_amb = _as_1d("t_amb_c", self.t_amb_c)
        leg = np.asarray(self.legionella_required, dtype=bool).flatten()

        n = v_tap.size
        for name, arr in (("t_mains_c", t_mains), ("t_amb_c", t_amb)):
            if arr.size != n:
                raise ValueError(f"{name} must have length {n}.")
        if leg.size != n:
            raise ValueError("legionella_required must have length N.")
        if np.any(v_tap < 0.0):
            raise ValueError("v_tap_m3_per_h cannot be negative.")

        # Time-varying COP: validate length and physical lower bound (> 1).
        if self.cop_dhw_k is not None:
            cop_arr = _as_1d("cop_dhw_k", self.cop_dhw_k)
            if cop_arr.size != n:
                raise ValueError(f"cop_dhw_k must have length {n}.")
            if np.any(cop_arr <= 1.0):
                raise ValueError("cop_dhw_k: all COP values must be > 1 (physical lower bound).")
            object.__setattr__(self, "cop_dhw_k", cop_arr)

        object.__setattr__(self, "v_tap_m3_per_h", v_tap)
        object.__setattr__(self, "t_mains_c", t_mains)
        object.__setattr__(self, "t_amb_c", t_amb)
        object.__setattr__(self, "legionella_required", leg)

    @property
    def horizon_steps(self) -> int:
        return int(self.v_tap_m3_per_h.size)

    @property
    def max_tap_flow_m3_per_h(self) -> float:
        """Return the maximum tap-flow disturbance over the horizon [m³/h]."""
        return float(np.max(self.v_tap_m3_per_h))

    def assert_compatible_with_parameters(
        self,
        parameters: DHWParameters,
        safety_factor: float = 0.2,
    ) -> None:
        """Fail fast when the forecast violates the DHW Euler stability bound.

        The MPC uses the DHW forecast as a known LTV disturbance sequence, so the
        physically relevant discretisation check must be based on the **largest**
        tap flow that appears anywhere in the horizon, not only the nominal scalar
        operating point.
        """
        parameters.assert_euler_stable_for_tap_flow(
            v_tap_m3_per_h=self.max_tap_flow_m3_per_h,
            safety_factor=safety_factor,
        )

    def disturbance_matrix(self) -> np.ndarray:
        """Return N×2 matrix with columns [T_amb, T_mains]."""
        return np.column_stack([self.t_amb_c, self.t_mains_c])

    @classmethod
    def constant(
        cls,
        horizon_steps: int,
        v_tap_m3_per_h: float = 0.0,
        t_mains_c: float = 10.0,
        t_amb_c: float = 20.0,
        legionella_required: bool = False,
    ) -> "DHWForecastHorizon":
        """Convenience factory: constant disturbances over the full horizon."""
        n = horizon_steps
        return cls(
            v_tap_m3_per_h=np.full(n, v_tap_m3_per_h),
            t_mains_c=np.full(n, t_mains_c),
            t_amb_c=np.full(n, t_amb_c),
            legionella_required=np.full(n, legionella_required, dtype=bool),
        )
