"""FastAPI web interface for the Home Optimizer.

Endpoints
---------
GET  /               HTML single-page application
GET  /api/defaults   Default RunRequest as JSON
POST /api/optimize   Run MPC, return Plotly chart JSON

Run with:
  uvicorn home_optimizer.api:app --reload
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

from .mpc import UFHMPCController
from .thermal_model import ThermalModel
from .types import ForecastHorizon, MPCParameters, ThermalParameters

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


def _solar_gti(hour: int) -> float:
    """Simple bell-shaped GTI [W/m²] centred at solar noon (13 h)."""
    if 7 <= hour <= 19:
        return 800.0 * np.sin(np.pi * (hour - 7) / 12.0)
    return 0.0


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """All adjustable parameters for one MPC solve."""

    # House thermal parameters
    C_r: float = Field(3.0, ge=0.5, le=50.0, description="Room thermal capacity [kWh/K]")
    C_b: float = Field(18.0, ge=1.0, le=200.0, description="Floor thermal capacity [kWh/K]")
    R_br: float = Field(2.5, ge=0.1, le=20.0, description="Floor→room resistance [K/kW]")
    R_ro: float = Field(4.0, ge=0.1, le=30.0, description="Room→outside resistance [K/kW]")
    alpha: float = Field(0.35, ge=0.0, le=1.0, description="Solar fraction to room air")

    # Initial state
    T_r_init: float = Field(20.8, ge=5.0, le=35.0, description="Initial room temperature [°C]")
    T_b_init: float = Field(24.0, ge=5.0, le=45.0, description="Initial floor temperature [°C]")
    previous_power_kw: float = Field(0.5, ge=0.0, le=20.0, description="Previous UFH power [kW]")

    # MPC settings
    horizon_hours: int = Field(24, ge=4, le=48, description="Horizon N [h, Δt = 1 h]")
    Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c")
    R_c: float = Field(0.05, ge=0.0, description="Regularisation weight R_c")
    P_max: float = Field(4.0, ge=0.5, le=20.0, description="Max UFH power [kW]")
    delta_P_max: float = Field(1.0, ge=0.1, le=10.0, description="Max ramp-rate [kW/step]")
    T_min: float = Field(19.0, ge=10.0, le=25.0, description="Min comfort temperature [°C]")
    T_max: float = Field(22.5, ge=16.0, le=30.0, description="Max comfort temperature [°C]")
    T_ref: float = Field(21.0, ge=15.0, le=26.0, description="Setpoint temperature [°C]")

    # Forecast
    outdoor_temperature_c: float = Field(
        10.0, ge=-20.0, le=35.0, description="Outdoor temperature [°C]"
    )
    dynamic_price: bool = Field(True, description="Use typical Dutch price pattern")
    flat_price: float = Field(0.25, ge=0.0, le=2.0, description="Flat price [€/kWh]")
    solar_gain: bool = Field(True, description="Include solar irradiance profile")
    internal_gains_kw: float = Field(0.35, ge=0.0, le=3.0, description="Internal gains [kW]")


class OptimizeResponse(BaseModel):
    status: str
    objective: float
    total_energy_kwh: float
    total_cost_eur: float
    first_power_kw: float
    max_comfort_violation_c: float
    temperature_fig: str  # Plotly figure JSON
    power_fig: str  # Plotly figure JSON


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------


def _build_forecast(req: RunRequest, start_hour: int) -> ForecastHorizon:
    N = req.horizon_hours
    hours = [(start_hour + k) % 24 for k in range(N)]

    prices = (
        np.array([_PRICES_24H[h] for h in hours])
        if req.dynamic_price
        else np.full(N, req.flat_price)
    )
    gti = np.array([_solar_gti(h) for h in hours], dtype=float) if req.solar_gain else np.zeros(N)
    return ForecastHorizon(
        outdoor_temperature_c=np.full(N, req.outdoor_temperature_c),
        gti_w_per_m2=gti,
        internal_gains_kw=np.full(N, req.internal_gains_kw),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(N + 1, req.T_ref),
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

    # ── Comfort band as a single closed polygon ───────────────────────────
    # Using fill="toself" on one trace avoids the ghost-line artefact that
    # occurs when two separate boundary traces (T_max / T_min) with width=0
    # inherit Plotly's default auto-colour (blue) and appear as extra lines.
    fig.add_trace(
        go.Scatter(
            x=labels + labels[::-1],
            y=[T_max] * n + [T_min] * n,
            fill="toself",
            fillcolor="rgba(100,149,237,0.18)",
            line=dict(color="rgba(0,0,0,0)"),  # completely transparent boundary
            name="Comfortband",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # ── Outdoor temperature ───────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_out] * n,
            name="T<sub>buiten</sub>",
            mode="lines",
            line=dict(color="#999", width=1.5, dash="dot"),
        )
    )

    # ── Comfort setpoint ──────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_ref] * n,
            name="T<sub>ref</sub> (setpoint)",
            mode="lines",
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
        )
    )

    # ── Predicted room temperature (single blue line) ─────────────────────
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=T_r,
            name="T<sub>r</sub> (kamer)",
            mode="lines+markers",
            line=dict(color="#1e6bbf", width=2.5),
            marker=dict(size=5),
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig.to_json()


def _power_figure(
    labels: list[str],
    P_UFH: np.ndarray,
    prices: np.ndarray,
    P_max: float,
) -> str:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # UFH power bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=P_UFH,
            name="P<sub>UFH</sub> [kW]",
            marker_color=[f"rgba(30,107,191,{0.5 + 0.5 * v / max(P_max, 0.01)})" for v in P_UFH],
        ),
        secondary_y=False,
    )

    # Electricity price line
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=prices,
            name="Prijs [€/kWh]",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hovermode="x unified",
        barmode="group",
    )
    fig.update_yaxes(
        title_text="Vermogen [kW]",
        secondary_y=False,
        range=[0, P_max * 1.1],
        gridcolor="#f5f5f5",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Prijs [€/kWh]",
        secondary_y=True,
        range=[0, max(prices) * 1.5],
        gridcolor=None,
        zeroline=False,
        showgrid=False,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Home Optimizer",
    description="UFH MPC dashboard",
    version="0.1.0",
)

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

    # Build physical model
    try:
        thermal_params = ThermalParameters(
            dt_hours=1.0,
            C_r=req.C_r,
            C_b=req.C_b,
            R_br=req.R_br,
            R_ro=req.R_ro,
            alpha=req.alpha,
            eta=0.62,
            A_glass=12.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mpc_params = MPCParameters(
        horizon_steps=req.horizon_hours,
        Q_c=req.Q_c,
        R_c=req.R_c,
        Q_N=req.Q_c * 1.5,
        P_max=req.P_max,
        delta_P_max=req.delta_P_max,
        T_min=req.T_min,
        T_max=req.T_max,
    )

    forecast = _build_forecast(req, start_hour)
    model = ThermalModel(thermal_params)
    controller = UFHMPCController(model=model, params=mpc_params)

    try:
        sol = controller.solve(
            initial_state_c=np.array([req.T_r_init, req.T_b_init]),
            forecast=forecast,
            previous_power_kw=req.previous_power_kw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    N = req.horizon_hours
    dt = thermal_params.dt_hours
    labels_states = _time_labels(start_hour, N + 1)
    labels_controls = _time_labels(start_hour, N)
    T_r = sol.predicted_states_c[:, 0]
    P_UFH = sol.control_sequence_kw
    prices = forecast.price_eur_per_kwh

    total_energy = float(np.sum(P_UFH) * dt)
    total_cost = float(np.sum(P_UFH * prices * dt))

    temp_fig = _temperature_figure(
        labels_states, T_r, req.T_ref, req.T_min, req.T_max, req.outdoor_temperature_c
    )
    power_fig = _power_figure(labels_controls, P_UFH, prices, req.P_max)

    return OptimizeResponse(
        status=sol.solver_status,
        objective=sol.objective_value,
        total_energy_kwh=total_energy,
        total_cost_eur=total_cost,
        first_power_kw=sol.first_control_kw,
        max_comfort_violation_c=sol.max_comfort_violation_c,
        temperature_fig=temp_fig,
        power_fig=power_fig,
    )
