"""Validated physical parameter dataclasses for UFH and DHW subsystems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import LAMBDA_WATER_KWH_PER_M3_K, LAMBDA_WATER_REFERENCE_TEMPERATURE_C


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
        for field_name in ("dt_hours", "C_r", "C_b", "R_br", "R_ro", "A_glass"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
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


@dataclass(frozen=True, slots=True)
class DHWParameters:
    """Physical parameters of the DHW 2-node stratification tank."""

    dt_hours: float
    C_top: float
    C_bot: float
    R_strat: float
    R_loss_top: float | None = None
    R_loss_bot: float | None = None
    R_loss: float | None = None
    heater_split_top: float = 0.0
    heater_split_bottom: float = 1.0
    lambda_water: float = LAMBDA_WATER_KWH_PER_M3_K
    lambda_water_reference_temperature_c: float = LAMBDA_WATER_REFERENCE_TEMPERATURE_C
    lambda_water_temperature_coefficient_per_k: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("dt_hours", "C_top", "C_bot", "R_strat", "lambda_water"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
        r_loss_top = self.R_loss_top
        r_loss_bot = self.R_loss_bot
        if r_loss_top is None and r_loss_bot is None:
            if self.R_loss is None or self.R_loss <= 0.0:
                raise ValueError(
                    "Provide positive R_loss_top/R_loss_bot or a positive legacy R_loss."
                )
            r_loss_top = self.R_loss
            r_loss_bot = self.R_loss
        elif r_loss_top is None or r_loss_bot is None:
            raise ValueError("R_loss_top and R_loss_bot must both be provided together.")
        if r_loss_top <= 0.0 or r_loss_bot <= 0.0:
            raise ValueError("R_loss_top and R_loss_bot must be strictly positive.")
        if not 0.0 <= self.heater_split_top <= 1.0:
            raise ValueError("heater_split_top must be in [0, 1].")
        if not 0.0 <= self.heater_split_bottom <= 1.0:
            raise ValueError("heater_split_bottom must be in [0, 1].")
        if not np.isclose(self.heater_split_top + self.heater_split_bottom, 1.0, atol=1e-9):
            raise ValueError("heater_split_top + heater_split_bottom must equal 1 within tolerance.")
        object.__setattr__(self, "R_loss_top", float(r_loss_top))
        object.__setattr__(self, "R_loss_bot", float(r_loss_bot))
        if self.R_loss is None:
            object.__setattr__(self, "R_loss", float((r_loss_top + r_loss_bot) / 2.0))
        if self.lambda_water_at_temperature_c(self.lambda_water_reference_temperature_c) <= 0.0:
            raise ValueError("lambda_water(T_reference) must remain strictly positive.")

    def lambda_water_at_temperature_c(self, temperature_c: float) -> float:
        """Return the volumetric water heat capacity λ(T) [kWh/(m³·K)].

        The default implementation is affine around a named reference temperature:

            λ(T) = λ_ref · (1 + k_λ · (T - T_ref))

        with:
        - ``λ_ref = lambda_water``
        - ``T_ref = lambda_water_reference_temperature_c``
        - ``k_λ = lambda_water_temperature_coefficient_per_k``

        Setting ``k_λ = 0`` recovers the legacy constant-property model exactly.
        """
        lambda_value = self.lambda_water * (
            1.0
            + self.lambda_water_temperature_coefficient_per_k
            * (temperature_c - self.lambda_water_reference_temperature_c)
        )
        if lambda_value <= 0.0:
            raise ValueError(
                f"lambda_water(T={temperature_c:.3f} °C) must remain strictly positive; got {lambda_value:.6g}."
            )
        return float(lambda_value)

    def lambda_water_temperature_derivative(self) -> float:
        """Return dλ/dT for the affine λ(T) law [kWh/(m³·K²)]."""
        return float(self.lambda_water * self.lambda_water_temperature_coefficient_per_k)

    @property
    def euler_time_constants_hours(self) -> tuple[float, float, float]:
        """Dominant time constants for the Euler stability criterion [h]."""
        return (
            self.C_top * self.R_strat,
            self.C_bot * self.R_strat,
            self.C_top * self.R_loss_top,
            self.C_bot * self.R_loss_bot,
        )

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
        """Return the DHW top-layer tap time constant ``C_top / (λ·V_tap)`` [h]."""
        if v_tap_m3_per_h < 0.0:
            raise ValueError("v_tap_m3_per_h must be non-negative.")
        if v_tap_m3_per_h == 0.0:
            return float("inf")
        lambda_water = self.lambda_water_at_temperature_c(self.lambda_water_reference_temperature_c)
        return self.C_top / (lambda_water * v_tap_m3_per_h)

    def max_stable_euler_dt_for_tap_flow(self, v_tap_m3_per_h: float, safety_factor: float = 0.2) -> float:
        """Return the most restrictive Euler bound [h] for the supplied tap flow."""
        base_limit = self.max_stable_euler_dt(safety_factor)
        tap_tau_hours = self.tap_flow_time_constant_hours(v_tap_m3_per_h)
        if np.isinf(tap_tau_hours):
            return base_limit
        return min(base_limit, safety_factor * tap_tau_hours)

    def assert_euler_stable_for_tap_flow(self, v_tap_m3_per_h: float, safety_factor: float = 0.2) -> None:
        """Raise when ``dt_hours`` violates the DHW Euler bound at a given tap flow."""
        limit = self.max_stable_euler_dt_for_tap_flow(v_tap_m3_per_h, safety_factor)
        if self.dt_hours > limit:
            raise ValueError(
                f"Forward-Euler time step dt={self.dt_hours:.3f} h exceeds the DHW stability "
                f"bound {limit:.3f} h at V_tap={v_tap_m3_per_h:.6f} m³/h. "
                "Reduce dt, reduce the admissible tap flow, or switch to ZOH discretisation."
            )


__all__ = ["DHWParameters", "ThermalParameters"]
