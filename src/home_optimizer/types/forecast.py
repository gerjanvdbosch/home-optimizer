"""Forecast horizon dataclasses for the UFH and DHW MPC models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .constants import W_PER_KW
from .physical import DHWParameters, ThermalParameters

Array1DInput = Iterable[float] | np.ndarray


def _as_1d(name: str, values: Array1DInput) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array-like.")
    return arr.copy()


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    """Disturbance forecast and reference trajectory over N steps."""

    outdoor_temperature_c: np.ndarray
    gti_w_per_m2: np.ndarray
    internal_gains_kw: np.ndarray
    price_eur_per_kwh: np.ndarray
    room_temperature_ref_c: np.ndarray
    feed_in_price_eur_per_kwh: np.ndarray | None = None
    pv_kw: np.ndarray | None = None
    cop_ufh_k: np.ndarray | None = None
    shutter_pct: np.ndarray | None = None

    def __post_init__(self) -> None:
        t_out = _as_1d("outdoor_temperature_c", self.outdoor_temperature_c)
        gti = _as_1d("gti_w_per_m2", self.gti_w_per_m2)
        q_int = _as_1d("internal_gains_kw", self.internal_gains_kw)
        price = _as_1d("price_eur_per_kwh", self.price_eur_per_kwh)
        t_ref = _as_1d("room_temperature_ref_c", self.room_temperature_ref_c)

        n = t_out.size
        for name, arr in (("gti_w_per_m2", gti), ("internal_gains_kw", q_int), ("price_eur_per_kwh", price)):
            if arr.size != n:
                raise ValueError(f"{name} must have length {n}.")
        if t_ref.size != n + 1:
            raise ValueError("room_temperature_ref_c must have length N+1 (includes terminal reference).")
        if np.any(gti < 0.0):
            raise ValueError("gti_w_per_m2 cannot be negative.")
        if np.any(price < 0.0):
            raise ValueError("price_eur_per_kwh cannot be negative.")

        if self.feed_in_price_eur_per_kwh is None:
            feed_in = np.zeros(n)
        else:
            feed_in = _as_1d("feed_in_price_eur_per_kwh", self.feed_in_price_eur_per_kwh)
            if feed_in.size != n:
                raise ValueError(f"feed_in_price_eur_per_kwh must have length {n}.")
            if np.any(feed_in < 0.0):
                raise ValueError("feed_in_price_eur_per_kwh cannot be negative.")

        if self.pv_kw is None:
            pv = np.zeros(n)
        else:
            pv = _as_1d("pv_kw", self.pv_kw)
            if pv.size != n:
                raise ValueError(f"pv_kw must have length {n}.")
            if np.any(pv < 0.0):
                raise ValueError("pv_kw cannot be negative.")

        if self.cop_ufh_k is not None:
            cop_arr = _as_1d("cop_ufh_k", self.cop_ufh_k)
            if cop_arr.size != n:
                raise ValueError(f"cop_ufh_k must have length {n}.")
            if np.any(cop_arr <= 1.0):
                raise ValueError("cop_ufh_k: all COP values must be > 1 (physical lower bound).")
            object.__setattr__(self, "cop_ufh_k", cop_arr)

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
        object.__setattr__(self, "feed_in_price_eur_per_kwh", feed_in)
        object.__setattr__(self, "room_temperature_ref_c", t_ref)
        object.__setattr__(self, "pv_kw", pv)

    @property
    def horizon_steps(self) -> int:
        return int(self.outdoor_temperature_c.size)

    def solar_gains_kw(self, parameters: ThermalParameters) -> np.ndarray:
        """Convert GTI forecast to solar gain [kW], accounting for shutter position."""
        if self.shutter_pct is None:
            eta_eff = parameters.eta
        else:
            eta_eff = parameters.eta * (self.shutter_pct / 100.0)
        return parameters.A_glass * self.gti_w_per_m2 * eta_eff / W_PER_KW

    def disturbance_matrix(self, parameters: ThermalParameters) -> np.ndarray:
        """Return N×3 matrix with columns [T_out, Q_solar, Q_int]."""
        return np.column_stack([
            self.outdoor_temperature_c,
            self.solar_gains_kw(parameters),
            self.internal_gains_kw,
        ])

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
            feed_in_price_eur_per_kwh=np.zeros(n),
            room_temperature_ref_c=np.full(n + 1, room_temperature_ref_c),
            pv_kw=np.full(n, pv_kw),
        )


@dataclass(frozen=True, slots=True)
class DHWForecastHorizon:
    """DHW disturbance forecast over N steps."""

    v_tap_m3_per_h: np.ndarray
    t_mains_c: np.ndarray
    t_amb_c: np.ndarray
    legionella_required: np.ndarray
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

    def assert_compatible_with_parameters(self, parameters: DHWParameters, safety_factor: float = 0.2) -> None:
        """Fail fast on non-finite DHW disturbance inputs for runtime discretisation."""
        if safety_factor <= 0.0:
            raise ValueError("safety_factor must be strictly positive.")
        if not np.isfinite(parameters.dt_hours):
            raise ValueError("parameters.dt_hours must be finite.")
        if not np.isfinite(self.max_tap_flow_m3_per_h):
            raise ValueError("v_tap_m3_per_h must be finite across the DHW horizon.")

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


__all__ = ["DHWForecastHorizon", "ForecastHorizon"]

