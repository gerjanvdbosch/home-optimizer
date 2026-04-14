"""FastAPI web interface for the Home Optimizer.

Endpoints
---------
GET  /               HTML single-page application
GET  /api/defaults   Default RunRequest as JSON
POST /api/optimize   Run MPC, return Plotly chart JSON
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
# Preset forecast data
# ---------------------------------------------------------------------------

# Typical Dutch day-ahead price pattern [€/kWh], hour 0–23
_PRICES_24H = np.array(
    [
        0.21,
        0.20,
        0.19,
        0.18,
        0.19,
        0.22,  # 00–05 (cheap night)
        0.28,
        0.35,
        0.38,
        0.36,
        0.32,
        0.28,  # 06–11 (morning peak)
        0.25,
        0.24,
        0.24,
        0.25,
        0.28,
        0.35,  # 12–17 (afternoon)
        0.42,
        0.45,
        0.40,
        0.35,
        0.28,
        0.23,  # 18–23 (evening peak)
    ],
    dtype=float,
)

# Solar proxy constants (bell-shaped irradiance profile)
_SOLAR_PEAK_W_PER_M2: float = 550.0   # peak irradiance at solar noon [W/m²]
_SOLAR_RISE_HOUR: int = 7              # proxy sunrise hour
_SOLAR_SET_HOUR: int = 19             # proxy sunset hour
_SOLAR_PERIOD_H: float = 12.0         # daylight duration for sine argument [h]


def _solar_gti(hour: int) -> float:
    """Bell-shaped south-facing solar proxy [W/m²] centred at solar noon."""
    if _SOLAR_RISE_HOUR <= hour <= _SOLAR_SET_HOUR:
        return _SOLAR_PEAK_W_PER_M2 * np.sin(
            np.pi * (hour - _SOLAR_RISE_HOUR) / _SOLAR_PERIOD_H
        )
    return 0.0


def _pv_generation(hour: int, peak_kw: float) -> float:
    """Bell-shaped PV generation proxy [kW] with configurable peak capacity."""
    if _SOLAR_RISE_HOUR <= hour <= _SOLAR_SET_HOUR:
        return peak_kw * np.sin(np.pi * (hour - _SOLAR_RISE_HOUR) / _SOLAR_PERIOD_H)
    return 0.0


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """All adjustable parameters for one MPC solve."""

    # ── UFH: house thermal parameters ────────────────────────────────────
    C_r: float = Field(6.0, ge=0.5, le=50.0, description="Room-zone thermal capacity [kWh/K]")
    C_b: float = Field(10.0, ge=1.0, le=200.0, description="UFH floor/slab thermal capacity [kWh/K]")
    R_br: float = Field(1.0, ge=0.1, le=20.0, description="Floor→room resistance [K/kW]")
    R_ro: float = Field(10.0, ge=0.1, le=30.0, description="Room→outside resistance [K/kW]")
    alpha: float = Field(0.25, ge=0.0, le=1.0, description="Solar fraction direct to room air [-]")
    eta: float = Field(0.55, ge=0.0, le=1.0, description="Solar transmittance of the glazing [-]")
    A_glass: float = Field(7.5, ge=0.5, le=40.0, description="South-facing glazing area [m²]")

    # ── UFH: initial state ────────────────────────────────────────────────
    T_r_init: float = Field(20.5, ge=5.0, le=35.0, description="Initial room temperature [°C]")
    T_b_init: float = Field(22.5, ge=5.0, le=45.0, description="Initial floor temperature [°C]")
    previous_power_kw: float = Field(0.8, ge=0.0, le=20.0, description="Previous UFH power [kW]")

    # ── MPC settings ──────────────────────────────────────────────────────
    horizon_hours: int = Field(24, ge=4, le=48, description="Horizon N [steps]")
    dt_hours: float = Field(1.0, ge=0.25, le=2.0, description="Time step Δt [h]")
    Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c")
    R_c: float = Field(0.05, ge=0.0, description="Regularisation weight R_c")
    Q_N: float = Field(12.0, ge=0.0, description="Terminal comfort weight Q_N")
    P_max: float = Field(4.5, ge=0.5, le=20.0, description="Max UFH power [kW]")
    delta_P_max: float = Field(1.0, ge=0.1, le=10.0, description="Max UFH ramp-rate [kW/step]")
    T_min: float = Field(19.0, ge=10.0, le=25.0, description="Min comfort temperature [°C]")
    T_max: float = Field(22.5, ge=16.0, le=30.0, description="Max comfort temperature [°C]")
    T_ref: float = Field(20.5, ge=15.0, le=26.0, description="Setpoint temperature [°C]")

    # ── UFH: forecast ─────────────────────────────────────────────────────
    outdoor_temperature_c: float = Field(6.0, ge=-20.0, le=35.0, description="Outdoor temperature [°C]")
    dynamic_price: bool = Field(True, description="Use typical Dutch day-ahead price pattern")
    flat_price: float = Field(0.25, ge=0.0, le=2.0, description="Flat price [€/kWh]")
    solar_gain: bool = Field(True, description="Include solar irradiance profile")
    internal_gains_kw: float = Field(0.30, ge=0.0, le=3.0, description="Internal gains [kW]")

    # ── PV forecast ───────────────────────────────────────────────────────
    pv_enabled: bool = Field(False, description="Enable PV self-consumption (reduces net grid cost)")
    pv_peak_kw: float = Field(4.0, ge=0.0, le=20.0, description="PV peak generation capacity [kW]")

    # ── DHW system ────────────────────────────────────────────────────────
    dhw_enabled: bool = Field(False, description="Enable DHW (domestic hot water) control")
    dhw_C_top: float = Field(0.5814, ge=0.01, le=5.0, description="DHW top-layer thermal capacity [kWh/K]")
    dhw_C_bot: float = Field(0.5814, ge=0.01, le=5.0, description="DHW bottom-layer thermal capacity [kWh/K]")
    dhw_R_strat: float = Field(10.0, ge=1.0, le=100.0, description="Stratification resistance [K/kW]")
    dhw_R_loss: float = Field(50.0, ge=5.0, le=200.0, description="Standby-loss resistance [K/kW]")
    dhw_T_top_init: float = Field(55.0, ge=20.0, le=85.0, description="Initial T_top [°C]")
    dhw_T_bot_init: float = Field(45.0, ge=15.0, le=80.0, description="Initial T_bot [°C]")
    dhw_P_max: float = Field(3.0, ge=0.5, le=15.0, description="Max DHW thermal power [kW]")
    dhw_delta_P_max: float = Field(1.0, ge=0.1, le=10.0, description="Max DHW ramp-rate [kW/step]")
    dhw_T_min: float = Field(50.0, ge=35.0, le=70.0, description="Min tap temperature [°C]")
    dhw_T_legionella: float = Field(60.0, ge=55.0, le=85.0, description="Legionella temperature [°C]")
    dhw_legionella_period_steps: int = Field(168, ge=24, le=336, description="Legionella cycle period [steps]")
    dhw_legionella_duration_steps: int = Field(1, ge=1, le=4, description="Legionella duration [steps]")
    dhw_v_tap_m3_per_h: float = Field(0.01, ge=0.0, le=0.2, description="Avg tap-water flow rate [m³/h]")
    dhw_t_mains_c: float = Field(10.0, ge=0.0, le=25.0, description="Mains water temperature [°C]")
    dhw_t_amb_c: float = Field(20.0, ge=5.0, le=35.0, description="Ambient temp around boiler [°C]")
    P_hp_max: float = Field(6.0, ge=1.0, le=30.0, description="Shared heat-pump max power [kW]")


class OptimizeResponse(BaseModel):
    status: str
    objective: float
    total_energy_kwh: float
    total_cost_eur: float
    first_power_kw: float
    max_comfort_violation_c: float
    # PV / grid
    pv_total_kwh: float = 0.0
    net_grid_energy_kwh: float = 0.0
    pv_enabled: bool = False
    control_labels: list[str]
    pv_forecast_kw: list[float]
    # DHW
    dhw_enabled: bool = False
    dhw_total_energy_kwh: float = 0.0
    dhw_total_cost_eur: float = 0.0
    max_dhw_comfort_violation_c: float = 0.0
    max_legionella_violation_c: float = 0.0
    # Charts
    temperature_fig: str
    power_fig: str
    pv_forecast_fig: str = ""
    dhw_fig: str = ""   # empty when DHW disabled


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------


def _build_ufh_forecast(req: RunRequest, start_hour: int) -> ForecastHorizon:
    N = req.horizon_hours
    hours = [(start_hour + k) % 24 for k in range(N)]
    prices = (
        np.array([_PRICES_24H[h] for h in hours])
        if req.dynamic_price
        else np.full(N, req.flat_price)
    )
    gti = np.array([_solar_gti(h) for h in hours], dtype=float) if req.solar_gain else np.zeros(N)
    pv = np.array([_pv_generation(h, req.pv_peak_kw) for h in hours], dtype=float) if req.pv_enabled else np.zeros(N)
    return ForecastHorizon(
        outdoor_temperature_c=np.full(N, req.outdoor_temperature_c),
        gti_w_per_m2=gti,
        internal_gains_kw=np.full(N, req.internal_gains_kw),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(N + 1, req.T_ref),
        pv_kw=pv,
    )


def _build_dhw_forecast(req: RunRequest, N: int) -> DHWForecastHorizon:
    return DHWForecastHorizon(
        v_tap_m3_per_h=np.full(N, req.dhw_v_tap_m3_per_h),
        t_mains_c=np.full(N, req.dhw_t_mains_c),
        t_amb_c=np.full(N, req.dhw_t_amb_c),
        legionella_required=np.zeros(N, dtype=bool),  # no legionella window in demo horizon
    )


def _time_labels(start_hour: int, n_points: int) -> list[str]:
    base = datetime.now(tz=timezone.utc).replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return [(base + timedelta(hours=k)).strftime("%H:%M") for k in range(n_points)]


def _temperature_figure(
    labels: list[str],
    T_r: np.ndarray,
    T_ref: float,
    T_min: float,
    T_max: float,
    T_out: float,
) -> str:
    fig = go.Figure()
    n = len(labels)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=[T_max] * n + [T_min] * n,
        fill="toself", fillcolor="rgba(100,149,237,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="Comfortband", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=labels, y=[T_out] * n, name="T<sub>buiten</sub>",
                             mode="lines", line=dict(color="#999", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=labels, y=[T_ref] * n, name="T<sub>ref</sub>",
                             mode="lines", line=dict(color="#2ca02c", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=labels, y=T_r, name="T<sub>r</sub> (kamer)",
                             mode="lines+markers", line=dict(color="#1e6bbf", width=2.5), marker=dict(size=5)))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig.to_json()


def _dhw_figure(labels: list[str], T_top: np.ndarray, T_bot: np.ndarray, T_dhw_min: float) -> str:
    fig = go.Figure()
    n = len(labels)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=[T_dhw_min] * n + [20.0] * n,
        fill="toself", fillcolor="rgba(255,165,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Comfort min", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=labels, y=T_top, name="T<sub>top</sub>",
                             mode="lines+markers", line=dict(color="#e74c3c", width=2.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=labels, y=T_bot, name="T<sub>bot</sub>",
                             mode="lines+markers", line=dict(color="#f39c12", width=2), marker=dict(size=4)))
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
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=P_UFH, name="P<sub>UFH</sub> [kW]",
        marker_color=[f"rgba(30,107,191,{0.5 + 0.5 * v / max(P_max, 0.01)})" for v in P_UFH],
    ), secondary_y=False)
    if np.any(P_dhw > 0):
        fig.add_trace(go.Bar(
            x=labels, y=P_dhw, name="P<sub>DHW</sub> [kW]",
            marker_color="rgba(231,76,60,0.65)",
        ), secondary_y=False)
    if np.any(pv_kw > 0):
        fig.add_trace(go.Scatter(
            x=labels, y=pv_kw, name="P<sub>PV</sub> [kW]",
            mode="lines", line=dict(color="#f1c40f", width=2, dash="dot"),
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=prices, name="Prijs [€/kWh]",
        mode="lines+markers", line=dict(color="#e74c3c", width=2), marker=dict(size=5),
    ), secondary_y=True)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hovermode="x unified", barmode="group",
    )
    fig.update_yaxes(title_text="Vermogen [kW]", secondary_y=False,
                     range=[0, P_max * 1.1], gridcolor="#f5f5f5", zeroline=False)
    fig.update_yaxes(title_text="Prijs [€/kWh]", secondary_y=True,
                     range=[0, max(prices) * 1.5], gridcolor=None, zeroline=False, showgrid=False)
    return fig.to_json()


def _pv_forecast_figure(labels: list[str], pv_kw: np.ndarray) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=pv_kw,
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

app = FastAPI(title="Home Optimizer", description="UFH + DHW MPC dashboard", version="0.2.0")
_TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(_TEMPLATE.read_text(encoding="utf-8"))


@app.get("/api/defaults")
async def defaults() -> RunRequest:
    return RunRequest()


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(req: RunRequest) -> OptimizeResponse:  # noqa: C901
    start_hour = datetime.now().hour
    N = req.horizon_hours

    # ── Build UFH thermal model ───────────────────────────────────────────
    try:
        thermal_params = ThermalParameters(
            dt_hours=req.dt_hours,
            C_r=req.C_r, C_b=req.C_b, R_br=req.R_br, R_ro=req.R_ro,
            alpha=req.alpha, eta=req.eta, A_glass=req.A_glass,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mpc_params = MPCParameters(
        horizon_steps=N, Q_c=req.Q_c, R_c=req.R_c, Q_N=req.Q_N,
        P_max=req.P_max, delta_P_max=req.delta_P_max,
        T_min=req.T_min, T_max=req.T_max,
    )

    ufh_forecast = _build_ufh_forecast(req, start_hour)
    ufh_model = ThermalModel(thermal_params)
    dt = thermal_params.dt_hours
    prices = ufh_forecast.price_eur_per_kwh
    pv_kw = ufh_forecast.pv_kw

    # ── Solve ─────────────────────────────────────────────────────────────
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
            )
            controller_params = CombinedMPCParameters(
                ufh=mpc_params, dhw=dhw_mpc_params, P_hp_max=req.P_hp_max,
            )
            dhw_model = DHWModel(dhw_params)
            dhw_forecast = _build_dhw_forecast(req, N)
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
        P_UFH = solution.ufh_control_sequence_kw
        P_dhw = solution.dhw_control_sequence_kw
        states = solution.predicted_states_c
        solver_status = solution.solver_status
        objective = solution.objective_value
        max_comfort_viol = solution.max_ufh_comfort_violation_c
        max_dhw_viol = solution.max_dhw_comfort_violation_c
        max_leg_viol = solution.max_legionella_violation_c

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── Compute energy / cost summaries ──────────────────────────────────
    P_hp_total = P_UFH + P_dhw
    P_import = np.maximum(P_hp_total - pv_kw, 0.0)
    total_energy = float(np.sum(P_hp_total) * dt)
    total_cost = float(np.sum(P_import * prices * dt))
    pv_total = float(np.sum(pv_kw) * dt)
    net_grid = float(np.sum(P_import) * dt)
    dhw_energy = float(np.sum(P_dhw) * dt)
    dhw_cost = float(np.sum(P_dhw * prices * dt))

    # ── Build charts ──────────────────────────────────────────────────────
    labels_states = _time_labels(start_hour, N + 1)
    labels_ctrl = _time_labels(start_hour, N)
    T_r = states[:, 0]

    temp_fig = _temperature_figure(labels_states, T_r, req.T_ref, req.T_min, req.T_max, req.outdoor_temperature_c)
    power_fig = _power_figure(labels_ctrl, P_UFH, P_dhw, pv_kw, prices, req.P_max)
    pv_forecast_fig = _pv_forecast_figure(labels_ctrl, pv_kw)
    dhw_fig = ""
    if req.dhw_enabled:
        T_top = states[:, 2]
        T_bot = states[:, 3]
        dhw_fig = _dhw_figure(labels_states, T_top, T_bot, req.dhw_T_min)

    return OptimizeResponse(
        status=solver_status,
        objective=objective,
        total_energy_kwh=total_energy,
        total_cost_eur=total_cost,
        first_power_kw=float(P_UFH[0]),
        max_comfort_violation_c=max_comfort_viol,
        pv_total_kwh=pv_total,
        net_grid_energy_kwh=net_grid,
        pv_enabled=req.pv_enabled,
        control_labels=labels_ctrl,
        pv_forecast_kw=pv_kw.tolist(),
        dhw_enabled=req.dhw_enabled,
        dhw_total_energy_kwh=dhw_energy,
        dhw_total_cost_eur=dhw_cost,
        max_dhw_comfort_violation_c=max_dhw_viol,
        max_legionella_violation_c=max_leg_viol,
        temperature_fig=temp_fig,
        power_fig=power_fig,
        pv_forecast_fig=pv_forecast_fig,
        dhw_fig=dhw_fig,
    )
