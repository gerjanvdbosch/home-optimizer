"""Validated parameter dataclasses and forecast containers.

Units
-----
Temperatures         : °C
Thermal capacity     : kWh/K
Thermal resistance   : K/kW
Power                : kW
Irradiance           : W/m²  (converted to kW inside ForecastHorizon)
Electricity price    : €/kWh
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Array1DInput = Iterable[float] | np.ndarray


def _as_1d(name: str, values: Array1DInput) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array-like.")
    return arr.copy()


# ---------------------------------------------------------------------------
# Physical / house parameters
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
        for field in ("dt_hours", "C_r", "C_b", "R_br", "R_ro", "A_glass"):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be strictly positive.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1].")

    # ------------------------------------------------------------------
    # Stability helpers
    # ------------------------------------------------------------------

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
# MPC parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MPCParameters:
    """Settings for the Model Predictive Controller.

    Parameters
    ----------
    horizon_steps:  Number of look-ahead steps N.
    Q_c:            Comfort weighting on (T_r - T_ref)² [dimensionless].
    R_c:            Regularisation weighting on P_UFH² (prevents power spikes).
    Q_N:            Terminal comfort weighting.
    P_max:          Maximum UFH power [kW].
    delta_P_max:    Maximum ramp-rate per step [kW/step].
    T_min:          Minimum allowed room temperature [°C].
    T_max:          Maximum allowed room temperature [°C].
    """

    horizon_steps: int
    Q_c: float
    R_c: float
    Q_N: float
    P_max: float
    delta_P_max: float
    T_min: float
    T_max: float

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be ≥ 1.")
        for field in ("Q_c", "R_c", "Q_N"):
            if getattr(self, field) < 0.0:
                raise ValueError(f"{field} must be non-negative.")
        for field in ("P_max", "delta_P_max"):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be strictly positive.")
        if self.T_min > self.T_max:
            raise ValueError("T_min must be ≤ T_max.")


# ---------------------------------------------------------------------------
# Kalman filter noise parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KalmanNoiseParameters:
    """Noise covariance parameters for the Kalman filter.

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
# Forecast horizon
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

        object.__setattr__(self, "outdoor_temperature_c", t_out)
        object.__setattr__(self, "gti_w_per_m2", gti)
        object.__setattr__(self, "internal_gains_kw", q_int)
        object.__setattr__(self, "price_eur_per_kwh", price)
        object.__setattr__(self, "room_temperature_ref_c", t_ref)

    @property
    def horizon_steps(self) -> int:
        return int(self.outdoor_temperature_c.size)

    def solar_gains_kw(self, p: ThermalParameters) -> np.ndarray:
        """Convert GTI forecast to solar gain [kW]: Q_solar = A_glass * GTI * η / 1000."""
        return p.A_glass * self.gti_w_per_m2 * p.eta / 1000.0

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
    ) -> "ForecastHorizon":
        """Convenience factory: constant disturbances over the full horizon."""
        n = horizon_steps
        return cls(
            outdoor_temperature_c=np.full(n, outdoor_temperature_c),
            gti_w_per_m2=np.full(n, gti_w_per_m2),
            internal_gains_kw=np.full(n, internal_gains_kw),
            price_eur_per_kwh=np.full(n, price_eur_per_kwh),
            room_temperature_ref_c=np.full(n + 1, room_temperature_ref_c),
        )
