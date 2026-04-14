"""FastAPI web interface for the Home Optimizer.

Architecture
------------
The API acts as the **integration layer** between the dashboard UI and the
thermal/MPC core.  A single ``POST /api/optimize`` endpoint:

1. Validates the user request via Pydantic (fail-fast on invalid physics).
2. Builds the Carnot COP model from the user's heat-pump parameters.
3. Computes time-varying COP arrays from the outdoor-temperature forecast
   (UFH via heating curve; DHW via fixed supply temperature).
4. Constructs the UFH and DHW forecast horizons, injecting the COP arrays.
5. Solves the MPC optimisation problem (CVXPY / greedy fallback).
6. Returns all results—numerical summaries + Plotly chart JSON—to the browser.

COP model integration (§14.1)
------------------------------
All costs and the shared electrical-budget constraint are computed on an
**electrical** power basis:

    P_elec[k] = P_thermal[k] / COP[k]

The COP arrays are pre-calculated using ``HeatPumpCOPModel`` *before* the MPC
is called.  This keeps the QP linear (COP is an exogenous, not endogenous,
quantity during the optimisation).

For UFH the COP varies via the heating curve (stooklijn):
    COP_UFH[k] = η · T_cond_K / (T_cond_K − T_evap_K)
    T_cond = T_aanvoer(T_out[k]) + Δ_cond   (aanvoertemp from heating curve)
    T_evap = T_out[k] − Δ_evap

For DHW the supply temperature is approximately the comfort setpoint T_dhw_min.

Endpoints
---------
GET  /               Single-page dashboard (HTML)
GET  /api/defaults   Default ``RunRequest`` as JSON
POST /api/optimize   Run one MPC step, return charts + numerical summaries
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

from .cop_model import HeatPumpCOPModel, HeatPumpCOPParameters
from .dhw_model import DHWModel
from .mpc import MPCController
from .thermal_model import ThermalModel
from .types import (
    CombinedMPCParameters,
    DHWForecastHorizon,
    DHWMPCParameters,
    DHWParameters,
    ForecastHorizon,
    MPCParameters,
    ThermalParameters,
)

# ---------------------------------------------------------------------------
# Named preset data — never hardcoded inline
# ---------------------------------------------------------------------------

#: Typical Dutch day-ahead electricity price pattern [€/kWh], hours 00–23.
_PRICES_24H = np.array(
    [
        0.21, 0.20, 0.19, 0.18, 0.19, 0.22,   # 00–05 cheap night
        0.28, 0.35, 0.38, 0.36, 0.32, 0.28,   # 06–11 morning peak
        0.25, 0.24, 0.24, 0.25, 0.28, 0.35,   # 12–17 afternoon
        0.42, 0.45, 0.40, 0.35, 0.28, 0.23,   # 18–23 evening peak
    ],
    dtype=float,
)

#: Peak south-facing irradiance at solar noon [W/m²].
_SOLAR_PEAK_W_PER_M2: float = 550.0
#: Hour at which the sun rises in the proxy model.
_SOLAR_RISE_HOUR: int = 7
#: Hour at which the sun sets in the proxy model.
_SOLAR_SET_HOUR: int = 19
#: Daylight duration used as the sine period argument [h].
_SOLAR_PERIOD_H: float = 12.0


def _solar_gti(hour: int) -> float:
    """Return the bell-shaped south-facing GTI proxy [W/m²] for a given hour.

    Models irradiance as a half-sine between sunrise and sunset, peaking at
    solar noon.  Used when no real forecast data is available.

    Args:
        hour: Hour of the day (0–23).

    Returns:
        Global Tilted Irradiance proxy [W/m²], ≥ 0.
    """
    if _SOLAR_RISE_HOUR <= hour <= _SOLAR_SET_HOUR:
        return _SOLAR_PEAK_W_PER_M2 * np.sin(
            np.pi * (hour - _SOLAR_RISE_HOUR) / _SOLAR_PERIOD_H
        )
    return 0.0


def _pv_generation(hour: int, peak_kw: float) -> float:
    """Return bell-shaped PV generation proxy [kW] for a given hour.

    Args:
        hour:    Hour of the day (0–23).
        peak_kw: PV system peak capacity [kW].

    Returns:
        Estimated PV output [kW], ≥ 0.
    """
    if _SOLAR_RISE_HOUR <= hour <= _SOLAR_SET_HOUR:
        return peak_kw * np.sin(
            np.pi * (hour - _SOLAR_RISE_HOUR) / _SOLAR_PERIOD_H
        )
    return 0.0


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """All user-adjustable parameters for one MPC optimisation step.

    The model uses Pydantic validation to enforce physical bounds on every
    parameter before they reach the solver.  Invalid values raise HTTP 422.
    """

    # ── UFH: two-zone house thermal model (§3–§5) ─────────────────────────
    C_r: float = Field(6.0, ge=0.5, le=50.0,
                       description="Room-air + furniture thermal capacity C_r [kWh/K]")
    C_b: float = Field(10.0, ge=1.0, le=200.0,
                       description="UFH floor / concrete slab thermal capacity C_b [kWh/K]")
    R_br: float = Field(1.0, ge=0.1, le=20.0,
                        description="Thermal resistance floor → room R_br [K/kW]")
    R_ro: float = Field(10.0, ge=0.1, le=30.0,
                        description="Thermal resistance room → outside R_ro [K/kW]")
    alpha: float = Field(0.25, ge=0.0, le=1.0,
                         description="Fraction of solar gain to room air α [-]")
    eta: float = Field(0.55, ge=0.0, le=1.0,
                       description="Window solar transmittance η [-]")
    A_glass: float = Field(7.5, ge=0.5, le=40.0,
                           description="South-facing glazing area A_glass [m²]")

    # ── UFH: initial conditions ───────────────────────────────────────────
    T_r_init: float = Field(20.5, ge=5.0, le=35.0,
                            description="Initial room-air temperature T_r [°C]")
    T_b_init: float = Field(22.5, ge=5.0, le=45.0,
                            description="Initial floor/slab temperature T_b [°C]")
    previous_power_kw: float = Field(0.8, ge=0.0, le=20.0,
                                     description="UFH power applied in previous step [kW]")

    # ── MPC settings (§14) ────────────────────────────────────────────────
    horizon_hours: int = Field(24, ge=4, le=48,
                               description="Horizon length N [steps]")
    dt_hours: float = Field(1.0, ge=0.25, le=2.0,
                            description="Forward-Euler time step Δt [h]")
    Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c [dimensionless]")
    R_c: float = Field(0.05, ge=0.0,
                       description="Regularisation weight R_c — damps power spikes")
    Q_N: float = Field(12.0, ge=0.0, description="Terminal comfort weight Q_N")
    P_max: float = Field(4.5, ge=0.5, le=20.0,
                         description="Max UFH **thermal** power P_UFH,max [kW]")
    delta_P_max: float = Field(1.0, ge=0.1, le=10.0,
                               description="Max UFH ramp-rate ΔP_UFH,max [kW/step]")
    T_min: float = Field(19.0, ge=10.0, le=25.0,
                         description="Minimum comfort temperature T_min [°C]")
    T_max: float = Field(22.5, ge=16.0, le=30.0,
                         description="Maximum comfort temperature T_max [°C]")
    T_ref: float = Field(20.5, ge=15.0, le=26.0,
                         description="Comfort setpoint T_ref [°C]")

    # ── UFH: disturbance forecast ─────────────────────────────────────────
    outdoor_temperature_c: float = Field(6.0, ge=-20.0, le=35.0,
                                         description="Outdoor temperature T_out [°C]")
    dynamic_price: bool = Field(True,
                                description="Use typical Dutch day-ahead price pattern")
    flat_price: float = Field(0.25, ge=0.0, le=2.0,
                              description="Flat electricity price p [€/kWh]")
    solar_gain: bool = Field(True, description="Include solar irradiance profile")
    internal_gains_kw: float = Field(0.30, ge=0.0, le=3.0,
                                     description="Internal heat gains Q_int [kW]")

    # ── PV self-consumption ───────────────────────────────────────────────
    pv_enabled: bool = Field(False,
                             description="Enable PV self-consumption (reduces net grid cost)")
    pv_peak_kw: float = Field(4.0, ge=0.0, le=20.0,
                              description="PV system peak capacity [kW]")

    # ── DHW: two-node stratification tank (§7–§11) ───────────────────────
    dhw_enabled: bool = Field(False,
                              description="Enable DHW (domestic hot water) control")
    dhw_C_top: float = Field(0.5814, ge=0.01, le=5.0,
                             description="DHW top-layer thermal capacity C_top [kWh/K]")
    dhw_C_bot: float = Field(0.5814, ge=0.01, le=5.0,
                             description="DHW bottom-layer thermal capacity C_bot [kWh/K]")
    dhw_R_strat: float = Field(10.0, ge=1.0, le=100.0,
                               description="Stratification resistance R_strat [K/kW]")
    dhw_R_loss: float = Field(50.0, ge=5.0, le=200.0,
                              description="Standby-loss resistance R_loss [K/kW]")
    dhw_T_top_init: float = Field(55.0, ge=20.0, le=85.0,
                                  description="Initial top-layer temperature T_top [°C]")
    dhw_T_bot_init: float = Field(45.0, ge=15.0, le=80.0,
                                  description="Initial bottom-layer temperature T_bot [°C]")
    dhw_P_max: float = Field(3.0, ge=0.5, le=15.0,
                             description="Max DHW **thermal** power P_dhw,max [kW]")
    dhw_delta_P_max: float = Field(1.0, ge=0.1, le=10.0,
                                   description="Max DHW ramp-rate ΔP_dhw,max [kW/step]")
    dhw_T_min: float = Field(50.0, ge=35.0, le=70.0,
                             description="Minimum tap (top-layer) temperature T_dhw,min [°C]")
    dhw_T_legionella: float = Field(60.0, ge=55.0, le=85.0,
                                    description="Legionella prevention temperature T_leg [°C]")
    dhw_legionella_period_steps: int = Field(
        168, ge=24, le=336,
        description="Legionella cycle period n_leg [steps]",
    )
    dhw_legionella_duration_steps: int = Field(
        1, ge=1, le=4,
        description="Min consecutive steps at T_legionella for legionella kill",
    )
    dhw_v_tap_m3_per_h: float = Field(0.01, ge=0.0, le=0.2,
                                      description="Average tap-water flow V̇_tap [m³/h]")
    dhw_t_mains_c: float = Field(10.0, ge=0.0, le=25.0,
                                 description="Cold mains-water temperature T_mains [°C]")
    dhw_t_amb_c: float = Field(20.0, ge=5.0, le=35.0,
                               description="Ambient temperature around the boiler T_amb [°C]")

    # ── Shared heat-pump electrical budget ───────────────────────────────
    P_hp_max_elec: float = Field(
        2.5, ge=0.5, le=30.0,
        description=(
            "Shared heat-pump **electrical** power budget P_hp,max,elec [kW]. "
            "Enforces P_UFH/COP_UFH + P_dhw/COP_dhw ≤ P_hp_max_elec (§14)."
        ),
    )

    # ── Warmtepomp – Carnot COP model (§14.1) ────────────────────────────
    eta_carnot: float = Field(
        0.45, ge=0.1, le=0.99,
        description=(
            "Carnot efficiency factor η [-].  Relates actual COP to the "
            "theoretical maximum: COP = η · T_cond_K / (T_cond_K − T_evap_K).  "
            "Typical air-source heat pump: 0.35–0.55."
        ),
    )
    delta_T_cond: float = Field(
        5.0, ge=0.0, le=15.0,
        description=(
            "Condensing approach temperature Δ_cond [K].  The refrigerant "
            "condenses at T_aanvoer + Δ_cond.  Typical: 2–8 K."
        ),
    )
    delta_T_evap: float = Field(
        5.0, ge=0.0, le=15.0,
        description=(
            "Evaporating approach temperature Δ_evap [K].  The refrigerant "
            "evaporates at T_buiten − Δ_evap.  Typical: 2–8 K."
        ),
    )
    T_supply_min: float = Field(
        28.0, ge=15.0, le=60.0,
        description=(
            "Minimum UFH supply temperature T_aanvoer,min [°C].  "
            "Floor of the heating curve (when T_out ≥ T_ref_outdoor_curve)."
        ),
    )
    T_ref_outdoor_curve: float = Field(
        18.0, ge=5.0, le=25.0,
        description=(
            "Balance-point outdoor temperature T_ref,buiten [°C].  "
            "At T_out = T_ref_outdoor_curve the heating curve equals T_supply_min."
        ),
    )
    heating_curve_slope: float = Field(
        1.0, ge=0.0, le=3.0,
        description=(
            "Stooklijn slope [K/K].  How much the supply temperature rises "
            "per K drop in outdoor temperature.  Typical UFH: 0.5–1.5."
        ),
    )
    cop_min: float = Field(
        1.5, ge=1.01, le=5.0,
        description="Physical lower bound on COP [-].  Must be > 1 (heat pump).",
    )
    cop_max: float = Field(
        7.0, ge=2.0, le=15.0,
        description="Upper bound on COP for Fail-Fast validation [-].",
    )


class OptimizeResponse(BaseModel):
    """Structured optimisation response for the dashboard and API clients.

    All monetary values are in EUR; energy in kWh (thermal or electrical as
    annotated); power in kW.
    """

    # ── Solver metadata ───────────────────────────────────────────────────
    status: str
    objective: float

    # ── Energy and cost summaries ─────────────────────────────────────────
    hp_total_energy_kwh: float         # thermal energy delivered [kWh therm]
    total_cost_eur: float              # electricity cost based on net grid import [€]
    ufh_total_energy_kwh: float        # UFH thermal energy [kWh therm]
    dhw_total_energy_kwh: float = 0.0  # DHW thermal energy [kWh therm]
    ufh_grid_cost_eur: float           # cost attributed to UFH by electrical share [€]
    dhw_grid_cost_eur: float = 0.0     # cost attributed to DHW by electrical share [€]
    first_ufh_power_kw: float          # first step UFH thermal power [kW]
    first_dhw_power_kw: float = 0.0    # first step DHW thermal power [kW]
    first_total_hp_power_kw: float     # first step total thermal power [kW]
    max_ufh_comfort_violation_c: float

    # ── PV / grid summaries ───────────────────────────────────────────────
    pv_total_kwh: float = 0.0
    net_grid_energy_kwh: float = 0.0
    pv_enabled: bool = False
    control_labels: list[str]
    pv_forecast_kw: list[float]

    # ── DHW summaries ─────────────────────────────────────────────────────
    dhw_enabled: bool = False
    max_dhw_comfort_violation_c: float = 0.0
    max_legionella_violation_c: float = 0.0

    # ── COP profile (§14.1) ───────────────────────────────────────────────
    cop_ufh_profile: list[float] = []  # UFH COP per time step over horizon
    cop_dhw_profile: list[float] = []  # DHW COP per time step (empty when DHW off)

    # ── Plotly chart JSON strings ─────────────────────────────────────────
    temperature_fig: str
    power_fig: str
    cop_fig: str = ""          # Carnot COP profile chart
    pv_forecast_fig: str = ""
    dhw_fig: str = ""          # empty when DHW disabled


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------


def _build_cop_model(req: RunRequest) -> HeatPumpCOPModel:
    """Construct the Carnot COP model from the user request.

    Assembles a ``HeatPumpCOPParameters`` dataclass from the request fields
    and wraps it in a ``HeatPumpCOPModel``.  Validation (e.g. cop_min > 1,
    eta_carnot ∈ (0,1]) is handled by ``HeatPumpCOPParameters.__post_init__``.

    Args:
        req: Validated Pydantic request containing all COP model parameters.

    Returns:
        Constructed and validated ``HeatPumpCOPModel``.

    Raises:
        ValueError: If any parameter violates a physical constraint.
    """
    params = HeatPumpCOPParameters(
        eta_carnot=req.eta_carnot,
        delta_T_cond=req.delta_T_cond,
        delta_T_evap=req.delta_T_evap,
        T_supply_min=req.T_supply_min,
        T_ref_outdoor=req.T_ref_outdoor_curve,
        heating_curve_slope=req.heating_curve_slope,
        cop_min=req.cop_min,
        cop_max=req.cop_max,
    )
    return HeatPumpCOPModel(params)


def _build_ufh_forecast(
    req: RunRequest,
    start_hour: int,
    cop_model: HeatPumpCOPModel,
) -> ForecastHorizon:
    """Build the UFH disturbance and price forecast over the horizon.

    Injects the time-varying UFH COP array (computed from the Carnot model
    and outdoor temperature) so the MPC can use physical electricity costs.

    Args:
        req:        Validated request with all forecast parameters.
        start_hour: Current hour of the day (0–23), used to index the price
                    pattern and solar profile.
        cop_model:  Carnot COP model; used to compute ``cop_ufh_k`` array.

    Returns:
        ``ForecastHorizon`` with N steps, including the COP array.
    """
    N = req.horizon_hours
    hours = [(start_hour + k) % 24 for k in range(N)]
    prices = (
        np.array([_PRICES_24H[h] for h in hours])
        if req.dynamic_price
        else np.full(N, req.flat_price)
    )
    gti = (
        np.array([_solar_gti(h) for h in hours], dtype=float)
        if req.solar_gain
        else np.zeros(N)
    )
    pv = (
        np.array([_pv_generation(h, req.pv_peak_kw) for h in hours], dtype=float)
        if req.pv_enabled
        else np.zeros(N)
    )
    t_out_arr = np.full(N, req.outdoor_temperature_c)
    # §14.1: Time-varying COP from the Carnot model + heating curve.
    # Colder outdoor → higher T_supply (heating curve) AND lower T_evap → lower COP.
    cop_ufh_k = cop_model.cop_ufh(t_out_arr)
    return ForecastHorizon(
        outdoor_temperature_c=t_out_arr,
        gti_w_per_m2=gti,
        internal_gains_kw=np.full(N, req.internal_gains_kw),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(N + 1, req.T_ref),
        pv_kw=pv,
        cop_ufh_k=cop_ufh_k,
    )


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
        ``DHWForecastHorizon`` with N steps, including the COP array.
    """
    t_out_arr = np.full(N, req.outdoor_temperature_c)
    # DHW supply temperature ≈ comfort setpoint T_dhw_min (normal operation).
    # During a legionella cycle the effective supply temp would be T_legionella;
    # the legionella scheduler adjusts the constraint, not the COP model here.
    cop_dhw_k = cop_model.cop_dhw(t_out_arr, t_dhw_supply=req.dhw_T_min)
    return DHWForecastHorizon(
        v_tap_m3_per_h=np.full(N, req.dhw_v_tap_m3_per_h),
        t_mains_c=np.full(N, req.dhw_t_mains_c),
        t_amb_c=np.full(N, req.dhw_t_amb_c),
        legionella_required=np.zeros(N, dtype=bool),  # scheduler handles this
        cop_dhw_k=cop_dhw_k,
    )


def _time_labels(start_hour: int, n_points: int) -> list[str]:
    """Generate HH:MM time labels starting at *start_hour* for *n_points* steps.

    Args:
        start_hour: Starting hour (0–23).
        n_points:   Number of labels to generate.

    Returns:
        List of ``"HH:MM"`` strings, one per time step.
    """
    base = datetime.now(tz=timezone.utc).replace(
        hour=start_hour, minute=0, second=0, microsecond=0
    )
    return [(base + timedelta(hours=k)).strftime("%H:%M") for k in range(n_points)]


def _temperature_figure(
    labels: list[str],
    T_r: np.ndarray,
    T_ref: float,
    T_min: float,
    T_max: float,
    T_out: float,
) -> str:
    """Build the UFH room-temperature Plotly figure.

    Shows the predicted room-temperature trajectory, the comfort band
    [T_min, T_max], the setpoint T_ref, and the (constant) outdoor temperature.

    Args:
        labels: Time axis labels, length N+1.
        T_r:    Predicted room-temperature trajectory [°C], shape (N+1,).
        T_ref:  Comfort setpoint [°C].
        T_min:  Lower comfort bound [°C].
        T_max:  Upper comfort bound [°C].
        T_out:  Outdoor temperature (constant over horizon) [°C].

    Returns:
        Plotly figure serialised to JSON string.
    """
    fig = go.Figure()
    n = len(labels)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=[T_max] * n + [T_min] * n,
        fill="toself", fillcolor="rgba(100,149,237,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="Comfortband", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[T_out] * n, name="T<sub>buiten</sub>",
        mode="lines", line=dict(color="#999", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[T_ref] * n, name="T<sub>ref</sub>",
        mode="lines", line=dict(color="#2ca02c", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=T_r, name="T<sub>r</sub> (kamer)",
        mode="lines+markers", line=dict(color="#1e6bbf", width=2.5),
        marker=dict(size=5),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig.to_json()


def _dhw_figure(
    labels: list[str],
    T_top: np.ndarray,
    T_bot: np.ndarray,
    T_dhw_min: float,
) -> str:
    """Build the DHW tank-temperature Plotly figure.

    Shows the predicted top-layer (tap-water outlet) and bottom-layer
    (heat-pump inlet) temperature trajectories, plus the comfort minimum.

    Args:
        labels:    Time axis labels, length N+1.
        T_top:     Predicted top-layer temperature [°C], shape (N+1,).
        T_bot:     Predicted bottom-layer temperature [°C], shape (N+1,).
        T_dhw_min: Minimum tap temperature for comfort [°C].

    Returns:
        Plotly figure serialised to JSON string.
    """
    fig = go.Figure()
    n = len(labels)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=[T_dhw_min] * n + [20.0] * n,
        fill="toself", fillcolor="rgba(255,165,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Comfort min", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=T_top, name="T<sub>top</sub>",
        mode="lines+markers", line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=T_bot, name="T<sub>bot</sub>",
        mode="lines+markers", line=dict(color="#f39c12", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig.to_json()


def _power_figure(
    labels: list[str],
    P_UFH: np.ndarray,
    P_dhw: np.ndarray,
    pv_kw: np.ndarray,
    prices: np.ndarray,
    P_max: float,
) -> str:
    """Build the heat-pump power + PV + electricity-price Plotly figure.

    Primary y-axis: UFH and DHW thermal power bars + PV generation line.
    Secondary y-axis: electricity price.

    Args:
        labels: Time axis labels, length N.
        P_UFH:  UFH thermal power sequence [kW], shape (N,).
        P_dhw:  DHW thermal power sequence [kW], shape (N,).
        pv_kw:  PV generation forecast [kW], shape (N,).
        prices: Electricity prices [€/kWh], shape (N,).
        P_max:  Maximum UFH thermal power [kW] — sets y-axis range.

    Returns:
        Plotly figure serialised to JSON string.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=P_UFH, name="P<sub>UFH</sub> [kW therm]",
        marker_color=[
            f"rgba(30,107,191,{0.5 + 0.5 * v / max(P_max, 0.01)})"
            for v in P_UFH
        ],
    ), secondary_y=False)
    if np.any(P_dhw > 0):
        fig.add_trace(go.Bar(
            x=labels, y=P_dhw, name="P<sub>DHW</sub> [kW therm]",
            marker_color="rgba(231,76,60,0.65)",
        ), secondary_y=False)
    if np.any(pv_kw > 0):
        fig.add_trace(go.Scatter(
            x=labels, y=pv_kw, name="P<sub>PV</sub> [kW]",
            mode="lines", line=dict(color="#f1c40f", width=2, dash="dot"),
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=prices, name="Prijs [€/kWh]",
        mode="lines+markers", line=dict(color="#e74c3c", width=2),
        marker=dict(size=5),
    ), secondary_y=True)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hovermode="x unified", barmode="group",
    )
    fig.update_yaxes(title_text="Thermisch vermogen [kW]", secondary_y=False,
                     range=[0, P_max * 1.1], gridcolor="#f5f5f5", zeroline=False)
    fig.update_yaxes(title_text="Prijs [€/kWh]", secondary_y=True,
                     range=[0, max(prices) * 1.5], gridcolor=None,
                     zeroline=False, showgrid=False)
    return fig.to_json()


def _cop_figure(
    labels: list[str],
    cop_ufh: np.ndarray,
    cop_dhw: np.ndarray | None = None,
    cop_min: float = 1.0,
) -> str:
    """Build the Carnot COP profile Plotly figure.

    Visualises the time-varying COP arrays that the MPC actually used.  This
    directly shows the impact of the heating curve and outdoor temperature on
    the system efficiency.

    Args:
        labels:  Time axis labels, length N.
        cop_ufh: UFH COP array [dimensionless], shape (N,).
        cop_dhw: DHW COP array [dimensionless], shape (N,).  ``None`` → DHW off.
        cop_min: Physical lower bound for COP; drawn as a reference line.

    Returns:
        Plotly figure serialised to JSON string.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=cop_ufh,
        name="COP<sub>UFH</sub> (stooklijn + Carnot)",
        mode="lines+markers",
        line=dict(color="#1e6bbf", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(30,107,191,0.08)",
    ))
    if cop_dhw is not None:
        fig.add_trace(go.Scatter(
            x=labels, y=cop_dhw,
            name="COP<sub>DHW</sub> (vaste aanvoertemp)",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.06)",
        ))
    # Reference: COP = 1 represents a resistive heater (worst case)
    fig.add_hline(
        y=cop_min,
        line=dict(color="#aaa", width=1, dash="dot"),
        annotation_text=f"COP_min = {cop_min}",
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(
            title="COP [-]",
            gridcolor="#f5f5f5",
            zeroline=False,
            rangemode="tozero",
        ),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig.to_json()


def _pv_forecast_figure(labels: list[str], pv_kw: np.ndarray) -> str:
    """Build the PV generation forecast Plotly figure.

    Args:
        labels: Time axis labels, length N.
        pv_kw:  PV generation forecast [kW], shape (N,).

    Returns:
        Plotly figure serialised to JSON string.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=pv_kw,
        name="P<sub>PV</sub> forecast [kW]",
        mode="lines+markers",
        line=dict(color="#f1c40f", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(241,196,15,0.18)",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="PV forecast [kW]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Home Optimizer",
    description=(
        "Gecombineerd thermisch model (UFH + DHW) met MPC-optimalisatie, "
        "Carnot COP-model, Kalman-filter toestandschatting en PV self-consumption."
    ),
    version="0.3.0",
)
_TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the single-page dashboard HTML."""
    return HTMLResponse(_TEMPLATE.read_text(encoding="utf-8"))


@app.get("/api/defaults")
async def defaults() -> RunRequest:
    """Return the default ``RunRequest`` as JSON (useful for UI initialisation)."""
    return RunRequest()


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(req: RunRequest) -> OptimizeResponse:  # noqa: C901
    """Run one MPC optimisation step and return charts + numerical summaries.

    Processing pipeline:

    1. Validate physical parameters (Pydantic + custom assertions).
    2. Build Carnot COP model from heat-pump parameters.
    3. Compute time-varying COP arrays: UFH via heating curve, DHW via
       fixed supply temperature (≈ T_dhw_min).
    4. Construct UFH and DHW forecast horizons with COP arrays embedded.
    5. Solve the QP (CVXPY / OSQP) or fall back to greedy heuristic.
    6. Compute electrical energy and cost summaries (on electrical basis).
    7. Serialise all Plotly charts to JSON for the browser.

    Args:
        req: Validated request with all physical and MPC parameters.

    Returns:
        ``OptimizeResponse`` with numerical results and Plotly chart JSON.

    Raises:
        HTTPException 422: If any parameter is physically invalid.
    """
    start_hour = datetime.now().hour
    N = req.horizon_hours

    # ── 1. Build UFH thermal model ──────────────────────────────────────
    try:
        thermal_params = ThermalParameters(
            dt_hours=req.dt_hours,
            C_r=req.C_r, C_b=req.C_b, R_br=req.R_br, R_ro=req.R_ro,
            alpha=req.alpha, eta=req.eta, A_glass=req.A_glass,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── 2. Build Carnot COP model (§14.1) ──────────────────────────────
    try:
        cop_model = _build_cop_model(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── 3. Compute representative scalar COPs for MPCParameters validation
    # The MPC will override these with the time-varying arrays from the
    # forecast (cop_ufh_k), but MPCParameters still requires a scalar for
    # the Fail-Fast bounds check.  Use the COP at the configured outdoor temp.
    t_rep = np.array([req.outdoor_temperature_c])
    cop_ufh_scalar = float(cop_model.cop_ufh(t_rep)[0])
    cop_dhw_scalar = float(cop_model.cop_dhw(t_rep, req.dhw_T_min)[0])

    mpc_params = MPCParameters(
        horizon_steps=N, Q_c=req.Q_c, R_c=req.R_c, Q_N=req.Q_N,
        P_max=req.P_max, delta_P_max=req.delta_P_max,
        T_min=req.T_min, T_max=req.T_max,
        cop_ufh=cop_ufh_scalar,   # derived from Carnot model, not a magic number
        cop_max=req.cop_max,
    )

    # ── 4. Build forecasts with embedded COP arrays ─────────────────────
    ufh_forecast = _build_ufh_forecast(req, start_hour, cop_model)
    ufh_model = ThermalModel(thermal_params)
    dt = thermal_params.dt_hours
    prices = ufh_forecast.price_eur_per_kwh
    pv_kw = ufh_forecast.pv_kw

    # ── 5. Solve MPC ────────────────────────────────────────────────────
    try:
        dhw_model: DHWModel | None = None
        dhw_forecast: DHWForecastHorizon | None = None
        controller_params: MPCParameters | CombinedMPCParameters = mpc_params
        initial_dhw_state: np.ndarray | None = None

        if req.dhw_enabled:
            dhw_params = DHWParameters(
                dt_hours=req.dt_hours,
                C_top=req.dhw_C_top, C_bot=req.dhw_C_bot,
                R_strat=req.dhw_R_strat, R_loss=req.dhw_R_loss,
            )
            dhw_mpc_params = DHWMPCParameters(
                P_dhw_max=req.dhw_P_max, delta_P_dhw_max=req.dhw_delta_P_max,
                T_dhw_min=req.dhw_T_min, T_legionella=req.dhw_T_legionella,
                legionella_period_steps=req.dhw_legionella_period_steps,
                legionella_duration_steps=req.dhw_legionella_duration_steps,
                cop_dhw=cop_dhw_scalar,   # derived from Carnot model
                cop_max=req.cop_max,
            )
            controller_params = CombinedMPCParameters(
                ufh=mpc_params,
                dhw=dhw_mpc_params,
                P_hp_max_elec=req.P_hp_max_elec,
            )
            dhw_model = DHWModel(dhw_params)
            dhw_forecast = _build_dhw_forecast(req, N, cop_model)
            initial_dhw_state = np.array([req.dhw_T_top_init, req.dhw_T_bot_init])

        controller = MPCController(
            ufh_model=ufh_model,
            params=controller_params,
            dhw_model=dhw_model,
        )
        solution = controller.solve(
            initial_ufh_state_c=np.array([req.T_r_init, req.T_b_init]),
            ufh_forecast=ufh_forecast,
            initial_dhw_state_c=initial_dhw_state,
            dhw_forecast=dhw_forecast,
            previous_p_ufh_kw=req.previous_power_kw,
        )
        P_UFH = np.maximum(solution.ufh_control_sequence_kw, 0.0)
        P_dhw = np.maximum(solution.dhw_control_sequence_kw, 0.0)
        states = solution.predicted_states_c
        solver_status = solution.solver_status
        objective = solution.objective_value
        max_comfort_viol = solution.max_ufh_comfort_violation_c
        max_dhw_viol = solution.max_dhw_comfort_violation_c
        max_leg_viol = solution.max_legionella_violation_c

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── 6. Compute energy / cost summaries (electrical basis) ───────────
    # §14.1: costs are always on electrical basis (thermal / COP).
    # Retrieve the COP arrays that the MPC actually used.
    assert ufh_forecast.cop_ufh_k is not None  # always set by _build_ufh_forecast
    cop_ufh_arr = ufh_forecast.cop_ufh_k
    cop_dhw_arr = (
        dhw_forecast.cop_dhw_k
        if dhw_forecast is not None and dhw_forecast.cop_dhw_k is not None
        else np.ones(N)
    )

    P_UFH_elec = P_UFH / cop_ufh_arr          # electrical power UFH [kW]
    P_dhw_elec = P_dhw / cop_dhw_arr          # electrical power DHW [kW]
    P_hp_total_elec = P_UFH_elec + P_dhw_elec # total electrical demand [kW]
    P_hp_total_therm = P_UFH + P_dhw          # total thermal output [kW]
    P_import = np.maximum(P_hp_total_elec - pv_kw, 0.0)  # net grid import [kW]

    total_energy = float(np.sum(P_hp_total_therm) * dt)   # [kWh therm]
    total_cost_steps = P_import * prices * dt              # per-step cost [€]
    total_cost = float(np.sum(total_cost_steps))
    pv_total = float(np.sum(pv_kw) * dt)
    net_grid = float(np.sum(P_import) * dt)
    ufh_energy = float(np.sum(P_UFH) * dt)
    dhw_energy = float(np.sum(P_dhw) * dt)

    # Cost attribution by electrical share (proportional to actual electricity drawn)
    ufh_cost_weights = np.divide(
        P_UFH_elec, P_hp_total_elec,
        out=np.zeros_like(P_UFH_elec), where=P_hp_total_elec > 0.0,
    )
    dhw_cost_weights = np.divide(
        P_dhw_elec, P_hp_total_elec,
        out=np.zeros_like(P_dhw_elec), where=P_hp_total_elec > 0.0,
    )
    ufh_grid_cost = float(np.sum(total_cost_steps * ufh_cost_weights))
    dhw_grid_cost = float(np.sum(total_cost_steps * dhw_cost_weights))

    # ── 7. Build charts ──────────────────────────────────────────────────
    labels_states = _time_labels(start_hour, N + 1)
    labels_ctrl = _time_labels(start_hour, N)
    T_r = states[:, 0]

    temp_fig = _temperature_figure(
        labels_states, T_r, req.T_ref, req.T_min, req.T_max,
        req.outdoor_temperature_c,
    )
    power_fig = _power_figure(labels_ctrl, P_UFH, P_dhw, pv_kw, prices, req.P_max)
    cop_fig = _cop_figure(
        labels_ctrl,
        cop_ufh=cop_ufh_arr,
        cop_dhw=cop_dhw_arr if req.dhw_enabled else None,
        cop_min=req.cop_min,
    )
    pv_forecast_fig = _pv_forecast_figure(labels_ctrl, pv_kw)
    dhw_fig = ""
    if req.dhw_enabled:
        T_top = states[:, 2]
        T_bot = states[:, 3]
        dhw_fig = _dhw_figure(labels_states, T_top, T_bot, req.dhw_T_min)

    return OptimizeResponse(
        status=solver_status,
        objective=objective,
        hp_total_energy_kwh=total_energy,
        total_cost_eur=total_cost,
        ufh_total_energy_kwh=ufh_energy,
        dhw_total_energy_kwh=dhw_energy,
        ufh_grid_cost_eur=ufh_grid_cost,
        dhw_grid_cost_eur=dhw_grid_cost,
        first_ufh_power_kw=float(P_UFH[0]),
        first_dhw_power_kw=float(P_dhw[0]),
        first_total_hp_power_kw=float(P_hp_total_therm[0]),
        max_ufh_comfort_violation_c=max_comfort_viol,
        pv_total_kwh=pv_total,
        net_grid_energy_kwh=net_grid,
        pv_enabled=req.pv_enabled,
        control_labels=labels_ctrl,
        pv_forecast_kw=pv_kw.tolist(),
        dhw_enabled=req.dhw_enabled,
        max_dhw_comfort_violation_c=max_dhw_viol,
        max_legionella_violation_c=max_leg_viol,
        cop_ufh_profile=cop_ufh_arr.tolist(),
        cop_dhw_profile=cop_dhw_arr.tolist() if req.dhw_enabled else [],
        temperature_fig=temp_fig,
        power_fig=power_fig,
        cop_fig=cop_fig,
        pv_forecast_fig=pv_forecast_fig,
        dhw_fig=dhw_fig,
    )
