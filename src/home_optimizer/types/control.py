"""Validated MPC parameter dataclasses for UFH, DHW, and combined control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MPCParameters:
    """Settings for the UFH Model Predictive Controller."""

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
    P_min: float = 0.0
    on_off_control_enabled: bool = False
    switch_penalty_eur: float = 0.0
    rho_factor: float = 1000.0

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be ≥ 1.")
        for field_name in ("Q_c", "R_c", "Q_N"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in ("P_max", "delta_P_max"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
        if self.P_min < 0.0:
            raise ValueError("P_min must be non-negative.")
        if self.P_min > self.P_max:
            raise ValueError("P_min must be less than or equal to P_max.")
        if self.T_min > self.T_max:
            raise ValueError("T_min must be ≤ T_max.")
        if self.rho_factor <= 0.0:
            raise ValueError("rho_factor must be strictly positive.")
        if self.switch_penalty_eur < 0.0:
            raise ValueError("switch_penalty_eur must be non-negative.")
        if self.cop_max <= 1.0:
            raise ValueError("cop_max must be strictly greater than 1.")
        if self.cop_ufh <= 1.0:
            raise ValueError(f"cop_ufh={self.cop_ufh} is physically impossible: COP must be > 1.")
        if self.cop_ufh > self.cop_max:
            raise ValueError(f"cop_ufh={self.cop_ufh} exceeds cop_max={self.cop_max}.")


@dataclass(frozen=True, slots=True)
class DHWMPCParameters:
    """MPC settings for the DHW subsystem."""

    P_dhw_max: float
    delta_P_dhw_max: float
    T_dhw_min: float
    T_dhw_target: float
    T_legionella: float
    legionella_period_steps: int
    legionella_duration_steps: int
    cop_dhw: float
    cop_max: float
    P_dhw_min: float = 0.0
    on_off_control_enabled: bool = False
    switch_penalty_eur: float = 0.0
    terminal_top_min: float | None = None
    target_rho_factor: float = 25.0
    comfort_rho_factor: float = 1000.0
    legionella_rho_factor: float = 1e6

    def __post_init__(self) -> None:
        for field_name in ("P_dhw_max", "delta_P_dhw_max"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be strictly positive.")
        if self.P_dhw_min < 0.0:
            raise ValueError("P_dhw_min must be non-negative.")
        if self.P_dhw_min > self.P_dhw_max:
            raise ValueError("P_dhw_min must be less than or equal to P_dhw_max.")
        if self.T_dhw_min <= 0.0:
            raise ValueError("T_dhw_min must be positive.")
        if self.T_dhw_target < self.T_dhw_min:
            raise ValueError("T_dhw_target must be greater than or equal to T_dhw_min.")
        if self.T_legionella <= self.T_dhw_min:
            raise ValueError("T_legionella must be strictly greater than T_dhw_min.")
        if self.legionella_period_steps <= 0:
            raise ValueError("legionella_period_steps must be >= 1.")
        if self.legionella_duration_steps <= 0:
            raise ValueError("legionella_duration_steps must be >= 1.")
        if self.comfort_rho_factor <= 0.0:
            raise ValueError("comfort_rho_factor must be strictly positive.")
        if self.target_rho_factor <= 0.0:
            raise ValueError("target_rho_factor must be strictly positive.")
        if self.legionella_rho_factor <= 0.0:
            raise ValueError("legionella_rho_factor must be strictly positive.")
        if self.switch_penalty_eur < 0.0:
            raise ValueError("switch_penalty_eur must be non-negative.")
        if self.cop_max <= 1.0:
            raise ValueError("cop_max must be strictly greater than 1.")
        if self.cop_dhw <= 1.0:
            raise ValueError(f"cop_dhw={self.cop_dhw} is physically impossible: COP must be > 1.")
        if self.cop_dhw > self.cop_max:
            raise ValueError(f"cop_dhw={self.cop_dhw} exceeds cop_max={self.cop_max}.")
        if self.terminal_top_min is None:
            object.__setattr__(self, "terminal_top_min", self.T_dhw_min)
        elif self.terminal_top_min <= 0.0:
            raise ValueError("terminal_top_min must be strictly positive when provided.")


@dataclass(frozen=True, slots=True)
class CombinedMPCParameters:
    """Parameters for the combined UFH + DHW Model Predictive Controller."""

    ufh: MPCParameters
    dhw: DHWMPCParameters
    P_hp_max_elec: float
    heat_pump_topology: Literal["shared", "exclusive"] = "shared"
    exclusive_active_mode: Literal["ufh", "dhw"] | None = None

    def __post_init__(self) -> None:
        if self.P_hp_max_elec <= 0.0:
            raise ValueError("P_hp_max_elec must be strictly positive.")
        if self.heat_pump_topology not in {"shared", "exclusive"}:
            raise ValueError("heat_pump_topology must be one of 'shared' or 'exclusive'.")
        if self.heat_pump_topology == "shared":
            if self.exclusive_active_mode is not None:
                raise ValueError("exclusive_active_mode must be omitted when heat_pump_topology='shared'.")
        elif self.exclusive_active_mode is not None and self.exclusive_active_mode not in {"ufh", "dhw"}:
            raise ValueError("exclusive_active_mode must be 'ufh' or 'dhw' when provided.")
        elif (
            self.exclusive_active_mode is None
            and not (self.ufh.on_off_control_enabled and self.dhw.on_off_control_enabled)
        ):
            raise ValueError(
                "exclusive topology without a fixed active mode requires on_off_control_enabled "
                "for both UFH and DHW so the MPC can schedule the modes explicitly."
            )


__all__ = ["CombinedMPCParameters", "DHWMPCParameters", "MPCParameters"]
