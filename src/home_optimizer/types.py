from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Array1DInput = Iterable[float] | np.ndarray


def _as_1d_float_array(name: str, values: Array1DInput) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array-like input.")
    return array.copy()


@dataclass(frozen=True, slots=True)
class ThermalParameters:
    dt_hours: float
    C_r: float
    C_b: float
    R_br: float
    R_ro: float
    alpha: float
    eta: float
    A_glass: float

    def __post_init__(self) -> None:
        for field_name in ("dt_hours", "C_r", "C_b", "R_br", "R_ro", "A_glass"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in the closed interval [0, 1].")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must lie in the closed interval [0, 1].")

    @property
    def euler_time_constants_hours(self) -> tuple[float, float, float]:
        return (
            self.C_r * self.R_br,
            self.C_b * self.R_br,
            self.C_r * self.R_ro,
        )

    def recommended_max_euler_dt_hours(self, safety_factor: float = 0.2) -> float:
        if not 0.0 < safety_factor < 1.0:
            raise ValueError("safety_factor must lie strictly between 0 and 1.")
        return safety_factor * min(self.euler_time_constants_hours)

    def assert_euler_stable(self, safety_factor: float = 0.2) -> None:
        max_dt = self.recommended_max_euler_dt_hours(safety_factor=safety_factor)
        if self.dt_hours > max_dt:
            raise ValueError(
                "Forward Euler time step is too large for the thermal time constants: "
                f"dt={self.dt_hours:.3f} h exceeds {max_dt:.3f} h. "
                "Use a smaller dt or switch to ZOH discretization."
            )


@dataclass(frozen=True, slots=True)
class MPCParameters:
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
            raise ValueError("horizon_steps must be at least 1.")
        for field_name in ("Q_c", "R_c", "Q_N"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in ("P_max", "delta_P_max"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
        if self.T_min > self.T_max:
            raise ValueError("T_min must be less than or equal to T_max.")


@dataclass(frozen=True, slots=True)
class KalmanNoiseParameters:
    process_covariance: np.ndarray
    measurement_variance: float

    def __post_init__(self) -> None:
        process_covariance = np.asarray(self.process_covariance, dtype=float)
        if process_covariance.shape == (2,):
            process_covariance = np.diag(process_covariance)
        if process_covariance.shape != (2, 2):
            raise ValueError("process_covariance must be a 2x2 matrix or a length-2 diagonal.")
        if not np.allclose(process_covariance, process_covariance.T):
            raise ValueError("process_covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(process_covariance)) < -1e-10:
            raise ValueError("process_covariance must be positive semi-definite.")
        if self.measurement_variance <= 0.0:
            raise ValueError("measurement_variance must be strictly positive.")
        object.__setattr__(self, "process_covariance", process_covariance)


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    outdoor_temperature_c: np.ndarray
    gti_w_per_m2: np.ndarray
    internal_gains_kw: np.ndarray
    price_eur_per_kwh: np.ndarray
    room_temperature_ref_c: np.ndarray

    def __post_init__(self) -> None:
        outdoor_temperature_c = _as_1d_float_array(
            "outdoor_temperature_c", self.outdoor_temperature_c
        )
        gti_w_per_m2 = _as_1d_float_array("gti_w_per_m2", self.gti_w_per_m2)
        internal_gains_kw = _as_1d_float_array("internal_gains_kw", self.internal_gains_kw)
        price_eur_per_kwh = _as_1d_float_array("price_eur_per_kwh", self.price_eur_per_kwh)
        room_temperature_ref_c = _as_1d_float_array(
            "room_temperature_ref_c", self.room_temperature_ref_c
        )

        horizon_steps = outdoor_temperature_c.size
        for name, array in (
            ("gti_w_per_m2", gti_w_per_m2),
            ("internal_gains_kw", internal_gains_kw),
            ("price_eur_per_kwh", price_eur_per_kwh),
        ):
            if array.size != horizon_steps:
                raise ValueError(f"{name} must have length {horizon_steps}.")
        if room_temperature_ref_c.size != horizon_steps + 1:
            raise ValueError(
                "room_temperature_ref_c must have length horizon_steps + 1 "
                "to provide the terminal MPC reference."
            )
        if np.any(gti_w_per_m2 < 0.0):
            raise ValueError("gti_w_per_m2 cannot be negative.")
        if np.any(price_eur_per_kwh < 0.0):
            raise ValueError("price_eur_per_kwh cannot be negative.")

        object.__setattr__(self, "outdoor_temperature_c", outdoor_temperature_c)
        object.__setattr__(self, "gti_w_per_m2", gti_w_per_m2)
        object.__setattr__(self, "internal_gains_kw", internal_gains_kw)
        object.__setattr__(self, "price_eur_per_kwh", price_eur_per_kwh)
        object.__setattr__(self, "room_temperature_ref_c", room_temperature_ref_c)

    @property
    def horizon_steps(self) -> int:
        return int(self.outdoor_temperature_c.size)

    def solar_gains_kw(self, parameters: ThermalParameters) -> np.ndarray:
        return parameters.A_glass * self.gti_w_per_m2 * parameters.eta / 1000.0

    def disturbance_matrix(self, parameters: ThermalParameters) -> np.ndarray:
        solar_kw = self.solar_gains_kw(parameters)
        return np.column_stack([self.outdoor_temperature_c, solar_kw, self.internal_gains_kw])

    @classmethod
    def from_constant(
        cls,
        horizon_steps: int,
        outdoor_temperature_c: float,
        gti_w_per_m2: float,
        internal_gains_kw: float,
        price_eur_per_kwh: float,
        room_temperature_ref_c: float,
    ) -> "ForecastHorizon":
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be at least 1.")
        return cls(
            outdoor_temperature_c=np.full(horizon_steps, outdoor_temperature_c, dtype=float),
            gti_w_per_m2=np.full(horizon_steps, gti_w_per_m2, dtype=float),
            internal_gains_kw=np.full(horizon_steps, internal_gains_kw, dtype=float),
            price_eur_per_kwh=np.full(horizon_steps, price_eur_per_kwh, dtype=float),
            room_temperature_ref_c=np.full(horizon_steps + 1, room_temperature_ref_c, dtype=float),
        )
