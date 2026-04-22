"""Application-layer request and result models for runtime optimization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from .request_projection import (
    DhwControlConfig,
    DhwForecastConfig,
    DhwPhysicalConfig,
    SharedHeatPumpConfig,
    UfhControlConfig,
    UfhForecastConfig,
    UfhPhysicalConfig,
)
from ..control.mpc import MPCSolution
from ..domain.heat_pump.cop import HeatPumpCOPParameters
from ..pricing import PriceConfig
from ..types.constants import LAMBDA_WATER_KWH_PER_M3_K
from ..types.forecast import DHWForecastHorizon, ForecastHorizon
from ..types.physical import DHWParameters, ThermalParameters


class RunRequest(BaseModel):
    """All user-adjustable parameters for one MPC optimisation step."""

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
            "PV power is derived as (gti_pv / 1000) * pv_peak_kw.  "
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

    pv_enabled: bool = Field(
        True, description="Enable PV self-consumption (reduces net grid cost)"
    )
    pv_peak_kw: float = Field(2.0, ge=0.0, le=20.0, description="PV system peak capacity [kW]")

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
    dhw_R_loss_top: float | None = Field(
        None, gt=0.0, description="Top-node DHW standby-loss resistance R_loss_top [K/kW]."
    )
    dhw_R_loss_bot: float | None = Field(
        None, gt=0.0, description="Bottom-node DHW standby-loss resistance R_loss_bot [K/kW]."
    )
    dhw_heater_split_top: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of DHW heating power injected into the top node [-]."
    )
    dhw_heater_split_bottom: float = Field(
        1.0, ge=0.0, le=1.0, description="Fraction of DHW heating power injected into the bottom node [-]."
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
    dhw_terminal_top_min: float | None = Field(
        None,
        ge=10.0,
        le=85.0,
        description="Optional terminal lower bound on DHW top-layer temperature at horizon end [degC].",
    )
    dhw_v_tap_forecast: list[float] | None = Field(
        None,
        description=(
            "Hourly DHW tap-flow forecast Vdot_tap [m³/h], length N. "
            "Provided by the DHW tap forecaster once enough history is available. "
            "DHW solve paths require an explicit horizon forecast; there is no hidden zero-demand default. "
            "Can also be set explicitly for simulation or testing."
        ),
    )
    dhw_t_mains_c: float = Field(
        10.0, ge=0.0, le=25.0, description="Cold mains-water temperature T_mains [degC]"
    )
    dhw_t_amb_c: float = Field(
        20.0, ge=5.0, le=35.0, description="Ambient temperature around the boiler T_amb [degC]"
    )

    P_hp_max_elec: float = Field(
        2.5,
        ge=0.5,
        le=30.0,
        description=(
            "Shared heat-pump electrical power budget P_hp,max,elec [kW]. "
            "Enforces P_UFH/COP_UFH + P_dhw/COP_dhw <= P_hp_max_elec (section 14)."
        ),
    )
    heat_pump_topology: str = Field(
        "shared",
        description="Heat-pump topology policy: shared, exclusive_ufh, or exclusive_dhw.",
    )

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

    @property
    def ufh_physical_config(self) -> UfhPhysicalConfig:
        return UfhPhysicalConfig(
            parameters=ThermalParameters(
                dt_hours=self.dt_hours,
                C_r=self.C_r,
                C_b=self.C_b,
                R_br=self.R_br,
                R_ro=self.R_ro,
                alpha=self.alpha,
                eta=self.eta,
                A_glass=self.A_glass,
            ),
            initial_state_c=np.array([self.T_r_init, self.T_b_init], dtype=float),
            room_temperature_bias_c=self.room_temperature_bias_c,
        )

    @property
    def ufh_control_config(self) -> UfhControlConfig:
        return UfhControlConfig(
            horizon_steps=self.horizon_hours,
            q_c=self.Q_c,
            r_c=self.R_c,
            q_n=self.Q_N,
            p_max_kw=self.P_max,
            delta_p_max_kw_per_step=self.delta_P_max,
            t_min_c=self.T_min,
            t_max_c=self.T_max,
            t_ref_c=self.T_ref,
            previous_power_kw=self.previous_power_kw,
        )

    @property
    def ufh_forecast_config(self) -> UfhForecastConfig:
        return UfhForecastConfig(
            horizon_steps=self.horizon_hours,
            outdoor_temperature_c=self.outdoor_temperature_c,
            t_out_forecast=self.t_out_forecast,
            gti_window_forecast=self.gti_window_forecast,
            shutter_living_room_pct=self.shutter_living_room_pct,
            shutter_forecast=self.shutter_forecast,
            gti_pv_forecast=self.gti_pv_forecast,
            price_config=self.price_config,
            pv_enabled=self.pv_enabled,
            pv_peak_kw=self.pv_peak_kw,
            baseload_forecast=self.baseload_forecast,
            internal_gains_kw=self.internal_gains_kw,
            internal_gains_forecast=self.internal_gains_forecast,
            internal_gains_heat_fraction=self.internal_gains_heat_fraction,
            room_temperature_ref_c=self.T_ref,
        )

    @property
    def dhw_physical_config(self) -> DhwPhysicalConfig:
        return DhwPhysicalConfig(
            enabled=self.dhw_enabled,
            parameters=DHWParameters(
                dt_hours=self.dt_hours,
                C_top=self.dhw_C_top,
                C_bot=self.dhw_C_bot,
                R_strat=self.dhw_R_strat,
                R_loss_top=self.dhw_R_loss_top if self.dhw_R_loss_top is not None else self.dhw_R_loss,
                R_loss_bot=self.dhw_R_loss_bot if self.dhw_R_loss_bot is not None else self.dhw_R_loss,
                heater_split_top=self.dhw_heater_split_top,
                heater_split_bottom=self.dhw_heater_split_bottom,
                lambda_water=self.dhw_lambda_water_kwh_per_m3k,
            ),
            initial_state_c=np.array([self.dhw_T_top_init, self.dhw_T_bot_init], dtype=float),
            top_temperature_bias_c=self.dhw_top_temperature_bias_c,
            bottom_temperature_bias_c=self.dhw_bottom_temperature_bias_c,
            boiler_ambient_bias_c=self.dhw_boiler_ambient_bias_c,
        )

    @property
    def dhw_control_config(self) -> DhwControlConfig:
        return DhwControlConfig(
            enabled=self.dhw_enabled,
            p_max_kw=self.dhw_P_max,
            delta_p_max_kw_per_step=self.dhw_delta_P_max,
            t_min_c=self.dhw_T_min,
            t_legionella_c=self.dhw_T_legionella,
            legionella_period_steps=self.dhw_legionella_period_steps,
            legionella_duration_steps=self.dhw_legionella_duration_steps,
            terminal_top_min_c=self.dhw_terminal_top_min,
        )

    @property
    def dhw_forecast_config(self) -> DhwForecastConfig:
        return DhwForecastConfig(
            horizon_steps=self.horizon_hours,
            outdoor_temperature_c=self.outdoor_temperature_c,
            t_out_forecast=self.t_out_forecast,
            v_tap_forecast_m3_per_h=self.dhw_v_tap_forecast,
            t_mains_c=self.dhw_t_mains_c,
            t_ambient_c=self.dhw_t_amb_c,
            t_dhw_min_c=self.dhw_T_min,
        )

    @property
    def shared_heat_pump_config(self) -> SharedHeatPumpConfig:
        return SharedHeatPumpConfig(
            cop_parameters=HeatPumpCOPParameters(
                eta_carnot=self.eta_carnot,
                delta_T_cond=self.delta_T_cond,
                delta_T_evap=self.delta_T_evap,
                T_supply_min=self.T_supply_min,
                T_ref_outdoor=self.T_ref_outdoor_curve,
                heating_curve_slope=self.heating_curve_slope,
                cop_min=self.cop_min,
                cop_max=self.cop_max,
            ),
            cop_max=self.cop_max,
            hp_max_electrical_power_kw=self.P_hp_max_elec,
            topology=self.heat_pump_topology,
        )


@dataclass(frozen=True, slots=True)
class MPCStepResult:
    """Numerical result of one MPC solve step (no charts, no web concerns)."""

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
    """Immutable snapshot of the latest successful scheduled MPC execution."""

    solved_at_utc: datetime
    request: RunRequest
    result: MPCStepResult


__all__ = [
    "MPCStepResult",
    "RunRequest",
    "ScheduledRunSnapshot",
]
