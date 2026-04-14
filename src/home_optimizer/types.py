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

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

Array1DInput = Iterable[float] | np.ndarray

# ---------------------------------------------------------------------------
# Universal unit-conversion constants (named, never hardcoded inline)
# ---------------------------------------------------------------------------

#: 1 kW = 1000 W  — used to convert W/m² irradiance to kW.
W_PER_KW: float = 1000.0


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
    lambda_water:   ρ·c_p of water [kWh/(m³·K)]; physical constant, default 1.1628.
    """

    dt_hours: float
    C_top: float
    C_bot: float
    R_strat: float
    R_loss: float
    lambda_water: float = 1.1628  # kWh/(m³·K) — ρ·c_p water

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
# Greedy solver configuration (tuning parameters for the fallback solver)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GreedySolverConfig:
    """Numerical tuning parameters for the greedy fallback solver.

    All values are dimensionless algorithmic settings, not physical constants.

    Parameters
    ----------
    lookahead_weight:   Multiplier on Q_N for next-step lookahead term.
    grid_divisor:       Divides delta_P_max to obtain the candidate grid step.
    min_grid_step_kw:   Floor on the grid step to prevent division-by-zero [kW].
    min_candidates:     Minimum number of power-level candidates per step.
    max_candidates:     Maximum number of power-level candidates per step.
    """

    lookahead_weight: float = 5.0
    grid_divisor: float = 10.0
    min_grid_step_kw: float = 0.01
    min_candidates: int = 21
    max_candidates: int = 51

    def __post_init__(self) -> None:
        if self.lookahead_weight < 0.0:
            raise ValueError("lookahead_weight must be non-negative.")
        if self.grid_divisor <= 0.0:
            raise ValueError("grid_divisor must be strictly positive.")
        if self.min_grid_step_kw <= 0.0:
            raise ValueError("min_grid_step_kw must be strictly positive.")
        if self.min_candidates < 2:
            raise ValueError("min_candidates must be >= 2.")
        if self.max_candidates < self.min_candidates:
            raise ValueError("max_candidates must be >= min_candidates.")


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
    P_max:           Maximum UFH power [kW].
    delta_P_max:     Maximum ramp-rate per step [kW/step].
    T_min:           Minimum allowed room temperature [°C].
    T_max:           Maximum allowed room temperature [°C].
    rho_factor:      Soft-constraint penalty multiplier: ρ = rho_factor × max(Q_c, 1).
    greedy:          Numerical tuning for the greedy fallback solver.
    """

    horizon_steps: int
    Q_c: float
    R_c: float
    Q_N: float
    P_max: float
    delta_P_max: float
    T_min: float
    T_max: float
    rho_factor: float = 1000.0
    greedy: GreedySolverConfig = field(default_factory=GreedySolverConfig)

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


# ---------------------------------------------------------------------------
# MPC parameters — DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DHWMPCParameters:
    """MPC settings for the DHW subsystem.

    Parameters
    ----------
    P_dhw_max:              Maximum thermal power to the bottom layer [kW].
    delta_P_dhw_max:        Maximum ramp-rate per step [kW/step].
    T_dhw_min:              Minimum tap (top-layer) temperature for comfort [°C].
    T_legionella:           Legionella prevention temperature [°C] (typically 60).
    legionella_period_steps:  Steps between mandatory legionella cycles.
    legionella_duration_steps: Minimum consecutive steps at T_legionella.
    comfort_rho_factor:     Soft-constraint penalty multiplier for T_top < T_dhw_min.
    legionella_rho_factor:  Soft-constraint penalty multiplier for T_top < T_legionella.
    """

    P_dhw_max: float
    delta_P_dhw_max: float
    T_dhw_min: float
    T_legionella: float
    legionella_period_steps: int
    legionella_duration_steps: int
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


# ---------------------------------------------------------------------------
# MPC parameters — Combined UFH + DHW
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CombinedMPCParameters:
    """Parameters for the combined UFH + DHW Model Predictive Controller.

    Parameters
    ----------
    ufh:        UFH-specific MPC parameters (horizon, weights, bounds).
    dhw:        DHW-specific MPC parameters (bounds, legionella requirements).
    P_hp_max:   Maximum total heat-pump output (UFH + DHW combined) [kW].
    """

    ufh: MPCParameters
    dhw: DHWMPCParameters
    P_hp_max: float

    def __post_init__(self) -> None:
        if self.P_hp_max <= 0.0:
            raise ValueError("P_hp_max must be strictly positive.")
        if self.ufh.horizon_steps != self.ufh.horizon_steps:  # future-proof placeholder
            pass


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
    """

    outdoor_temperature_c: np.ndarray
    gti_w_per_m2: np.ndarray
    internal_gains_kw: np.ndarray
    price_eur_per_kwh: np.ndarray
    room_temperature_ref_c: np.ndarray
    #: PV generation forecast [kW], length N.  ``None`` means no PV installed → zeros.
    pv_kw: np.ndarray | None = None

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
        """Convert GTI forecast to solar gain [kW]: Q_solar = A_glass * GTI * η / W_PER_KW."""
        return p.A_glass * self.gti_w_per_m2 * p.eta / W_PER_KW

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
    """

    v_tap_m3_per_h: np.ndarray
    t_mains_c: np.ndarray
    t_amb_c: np.ndarray
    legionella_required: np.ndarray

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

        object.__setattr__(self, "v_tap_m3_per_h", v_tap)
        object.__setattr__(self, "t_mains_c", t_mains)
        object.__setattr__(self, "t_amb_c", t_amb)
        object.__setattr__(self, "legionella_required", leg)

    @property
    def horizon_steps(self) -> int:
        return int(self.v_tap_m3_per_h.size)

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

