"""Explicit domain projections derived from the broad runtime RunRequest model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.heat_pump.cop import HeatPumpCOPParameters
from ..pricing import PriceConfig
from ..types.control import CombinedMPCParameters, DHWMPCParameters, MPCParameters
from ..types.physical import DHWParameters, ThermalParameters


@dataclass(frozen=True, slots=True)
class UfhPhysicalConfig:
    """UFH physical configuration and initial state derived from one runtime request."""

    parameters: ThermalParameters
    initial_state_c: np.ndarray
    room_temperature_bias_c: float


@dataclass(frozen=True, slots=True)
class UfhControlConfig:
    """UFH MPC configuration derived from one runtime request."""

    horizon_steps: int
    q_c: float
    r_c: float
    q_n: float
    p_max_kw: float
    p_min_kw: float
    delta_p_max_kw_per_step: float
    t_min_c: float
    t_max_c: float
    t_ref_c: float
    previous_power_kw: float
    on_off_control_enabled: bool
    switch_penalty_eur: float

    def to_mpc_parameters(self, *, cop_ufh: float, cop_max: float) -> MPCParameters:
        """Return validated UFH MPC parameters using a representative COP sample."""
        return MPCParameters(
            horizon_steps=self.horizon_steps,
            Q_c=self.q_c,
            R_c=self.r_c,
            Q_N=self.q_n,
            P_max=self.p_max_kw,
            P_min=self.p_min_kw,
            delta_P_max=self.delta_p_max_kw_per_step,
            T_min=self.t_min_c,
            T_max=self.t_max_c,
            cop_ufh=cop_ufh,
            cop_max=cop_max,
            on_off_control_enabled=self.on_off_control_enabled,
            switch_penalty_eur=self.switch_penalty_eur,
        )


@dataclass(frozen=True, slots=True)
class UfhForecastConfig:
    """UFH forecast inputs, tariffs, shutters, PV, and internal-gains settings."""

    horizon_steps: int
    outdoor_temperature_c: float
    t_out_forecast: list[float] | None
    gti_window_forecast: list[float] | None
    shutter_living_room_pct: float
    shutter_forecast: list[float] | None
    gti_pv_forecast: list[float] | None
    price_config: PriceConfig
    pv_enabled: bool
    pv_peak_kw: float
    baseload_forecast: list[float] | None
    internal_gains_kw: float
    internal_gains_forecast: list[float] | None
    internal_gains_heat_fraction: float
    room_temperature_ref_c: float


@dataclass(frozen=True, slots=True)
class DhwPhysicalConfig:
    """DHW physical configuration and initial state derived from one runtime request."""

    enabled: bool
    parameters: DHWParameters
    initial_state_c: np.ndarray
    top_temperature_bias_c: float
    bottom_temperature_bias_c: float
    boiler_ambient_bias_c: float


@dataclass(frozen=True, slots=True)
class DhwControlConfig:
    """DHW MPC configuration derived from one runtime request."""

    enabled: bool
    p_max_kw: float
    p_min_kw: float
    delta_p_max_kw_per_step: float
    t_min_c: float
    t_target_c: float
    t_legionella_c: float
    legionella_period_steps: int
    legionella_duration_steps: int
    terminal_top_min_c: float | None
    on_off_control_enabled: bool
    switch_penalty_eur: float
    target_rho_factor: float

    def to_mpc_parameters(self, *, cop_dhw: float, cop_max: float) -> DHWMPCParameters:
        """Return validated DHW MPC parameters using a representative COP sample."""
        return DHWMPCParameters(
            P_dhw_max=self.p_max_kw,
            P_dhw_min=self.p_min_kw,
            delta_P_dhw_max=self.delta_p_max_kw_per_step,
            T_dhw_min=self.t_min_c,
            T_dhw_target=self.t_target_c,
            T_legionella=self.t_legionella_c,
            legionella_period_steps=self.legionella_period_steps,
            legionella_duration_steps=self.legionella_duration_steps,
            cop_dhw=cop_dhw,
            cop_max=cop_max,
            on_off_control_enabled=self.on_off_control_enabled,
            switch_penalty_eur=self.switch_penalty_eur,
            terminal_top_min=self.terminal_top_min_c,
            target_rho_factor=self.target_rho_factor,
        )


@dataclass(frozen=True, slots=True)
class DhwForecastConfig:
    """DHW disturbance and comfort forecast inputs derived from one runtime request."""

    horizon_steps: int
    outdoor_temperature_c: float
    t_out_forecast: list[float] | None
    v_tap_forecast_m3_per_h: list[float] | None
    t_mains_c: float
    t_ambient_c: float
    t_dhw_min_c: float
    t_dhw_target_c: float
    schedule_enabled: bool
    schedule_start_hour_local: int
    schedule_duration_hours: int
    schedule_target_c: float
    preheat_lead_steps: int
    significant_tap_threshold_m3_per_h: float


@dataclass(frozen=True, slots=True)
class SharedHeatPumpConfig:
    """Heat-pump COP and shared electrical-budget settings."""

    cop_parameters: HeatPumpCOPParameters
    cop_max: float
    hp_max_electrical_power_kw: float
    topology: str = "shared"
    exclusive_active_mode: str | None = None


def build_combined_mpc_parameters(
    *,
    ufh_control: UfhControlConfig,
    dhw_control: DhwControlConfig,
    shared_heat_pump: SharedHeatPumpConfig,
    cop_ufh: float,
    cop_dhw: float,
) -> CombinedMPCParameters:
    """Return validated combined MPC parameters from explicit domain projections."""
    return CombinedMPCParameters(
        ufh=ufh_control.to_mpc_parameters(cop_ufh=cop_ufh, cop_max=shared_heat_pump.cop_max),
        dhw=dhw_control.to_mpc_parameters(cop_dhw=cop_dhw, cop_max=shared_heat_pump.cop_max),
        P_hp_max_elec=shared_heat_pump.hp_max_electrical_power_kw,
        heat_pump_topology=shared_heat_pump.topology,
        exclusive_active_mode=shared_heat_pump.exclusive_active_mode,
    )


__all__ = [
    "DhwControlConfig",
    "DhwForecastConfig",
    "DhwPhysicalConfig",
    "SharedHeatPumpConfig",
    "UfhControlConfig",
    "UfhForecastConfig",
    "UfhPhysicalConfig",
    "build_combined_mpc_parameters",
]
