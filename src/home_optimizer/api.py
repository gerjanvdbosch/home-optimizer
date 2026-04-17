"""FastAPI web interface for the Home Optimizer.

Architecture
------------
The API acts as the **presentation layer** — it owns HTTP endpoints, Pydantic
request/response models, and Plotly chart serialisation.  All domain logic
lives in the dedicated core modules:

* :class:`~home_optimizer.optimizer.Optimizer` — orchestrates the full MPC
  solve pipeline (thermal model → COP model → forecasts → CVXPY → summaries).
* :class:`~home_optimizer.optimizer.MPCStepResult` — immutable result object
  returned by :meth:`~home_optimizer.optimizer.Optimizer.solve`.

A single ``POST /api/simulate`` endpoint:

1. Validates the user request via Pydantic (fail-fast on invalid physics).
2. Delegates the numerical solve to ``Optimizer().solve(req)``.
3. Assembles Plotly chart JSON from the result and returns the full
   ``OptimizeResponse`` to the browser.

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
GET  /                  Home Assistant operational dashboard (HTML)
GET  /simulator         MPC parameter simulator (HTML)
GET  /api/defaults      Default ``RunRequest`` as JSON
GET  /api/forecast      Fetch OpenMeteo weather forecast as JSON
POST /api/simulate      Run one MPC step, return charts + numerical summaries
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from pydantic import BaseModel

from .database import Database
from .optimizer import (  # noqa: F401 – re-exported for scheduler
    MPCStepResult,
    Optimizer,
    RunRequest,
)
from .sensors.open_meteo import OpenMeteoClient
from .telemetry import TelemetryRepository

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class OptimizeResponse(BaseModel):
    """Structured optimisation response for the dashboard and API clients.

    All monetary values are in EUR; energy in kWh (thermal or electrical as
    annotated); power in kW.
    """

    # ── Solver metadata ───────────────────────────────────────────────────
    status: str
    objective: float

    # ── Energy and cost summaries ─────────────────────────────────────────
    hp_total_energy_kwh: float  # thermal energy delivered [kWh therm]
    total_cost_eur: float  # electricity cost based on net grid import [€]
    ufh_total_energy_kwh: float  # UFH thermal energy [kWh therm]
    dhw_total_energy_kwh: float = 0.0  # DHW thermal energy [kWh therm]
    ufh_grid_cost_eur: float  # cost attributed to UFH by electrical share [€]
    dhw_grid_cost_eur: float = 0.0  # cost attributed to DHW by electrical share [€]
    first_ufh_power_kw: float  # first step UFH thermal power [kW]
    first_dhw_power_kw: float = 0.0  # first step DHW thermal power [kW]
    first_total_hp_power_kw: float  # first step total thermal power [kW]
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
    cop_fig: str = ""  # Carnot COP profile chart
    pv_forecast_fig: str = ""
    dhw_fig: str = ""  # empty when DHW disabled


def _time_labels(start_hour: int, n_points: int) -> list[str]:
    """Generate HH:MM time labels starting at *start_hour* for *n_points* steps.

    Args:
        start_hour: Starting hour (0–23).
        n_points:   Number of labels to generate.

    Returns:
        List of ``"HH:MM"`` strings, one per time step.
    """
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
    fig.add_trace(
        go.Scatter(
            x=labels + labels[::-1],
            y=[T_max] * n + [T_min] * n,
            fill="toself",
            fillcolor="rgba(100,149,237,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Comfortband",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_out] * n,
            name="T<sub>buiten</sub>",
            mode="lines",
            line=dict(color="#999", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_ref] * n,
            name="T<sub>ref</sub>",
            mode="lines",
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
        )
    )
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
    fig.add_trace(
        go.Scatter(
            x=labels + labels[::-1],
            y=[T_dhw_min] * n + [20.0] * n,
            fill="toself",
            fillcolor="rgba(255,165,0,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Comfort min",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=T_top,
            name="T<sub>top</sub>",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2.5),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=T_bot,
            name="T<sub>bot</sub>",
            mode="lines+markers",
            line=dict(color="#f39c12", width=2),
            marker=dict(size=4),
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
    fig.add_trace(
        go.Bar(
            x=labels,
            y=P_UFH,
            name="P<sub>UFH</sub> [kW therm]",
            marker_color=[f"rgba(30,107,191,{0.5 + 0.5 * v / max(P_max, 0.01)})" for v in P_UFH],
        ),
        secondary_y=False,
    )
    if np.any(P_dhw > 0):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=P_dhw,
                name="P<sub>DHW</sub> [kW therm]",
                marker_color="rgba(231,76,60,0.65)",
            ),
            secondary_y=False,
        )
    if np.any(pv_kw > 0):
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=pv_kw,
                name="P<sub>PV</sub> [kW]",
                mode="lines",
                line=dict(color="#f1c40f", width=2, dash="dot"),
            ),
            secondary_y=False,
        )
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
        title_text="Thermisch vermogen [kW]",
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
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=cop_ufh,
            name="COP<sub>UFH</sub> (stooklijn + Carnot)",
            mode="lines+markers",
            line=dict(color="#1e6bbf", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(30,107,191,0.08)",
        )
    )
    if cop_dhw is not None:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=cop_dhw,
                name="COP<sub>DHW</sub> (vaste aanvoertemp)",
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.06)",
            )
        )
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
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
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=pv_kw,
            name="P<sub>PV</sub> forecast [kW]",
            mode="lines+markers",
            line=dict(color="#f1c40f", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(241,196,15,0.18)",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="PV forecast [kW]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
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
_TEMPLATE_DASHBOARD = Path(__file__).parent / "templates" / "dashboard.html"
_TEMPLATE_SIMULATOR = Path(__file__).parent / "templates" / "simulator.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the Home Assistant operational dashboard HTML."""
    return HTMLResponse(_TEMPLATE_DASHBOARD.read_text(encoding="utf-8"))


@app.get("/simulator", response_class=HTMLResponse, include_in_schema=False)
async def simulator() -> HTMLResponse:
    """Serve the MPC parameter simulator HTML."""
    return HTMLResponse(_TEMPLATE_SIMULATOR.read_text(encoding="utf-8"))


@app.get("/api/defaults")
async def defaults() -> RunRequest:
    """Return the default ``RunRequest`` as JSON (useful for UI initialisation)."""
    return RunRequest.model_validate({})


class ForecastResponse(BaseModel):
    """OpenMeteo forecast response for the Home Assistant dashboard.

    All arrays have length ``horizon_steps``.  Time labels are ISO-8601 UTC strings.
    """

    horizon_steps: int
    dt_hours: float
    valid_from: str  # ISO-8601 UTC string
    labels: list[str]  # HH:MM labels for charts
    outdoor_temperature_c: list[float]  # T_out [C]
    gti_w_per_m2: list[float]  # window GTI [W/m²]
    gti_pv_w_per_m2: list[float]  # PV panel GTI [W/m²], 0 when not configured
    # Plotly chart JSON strings
    temperature_forecast_fig: str
    solar_forecast_fig: str


def _build_forecast_figures(
    labels: list[str],
    temps: list[float],
    gti_window: list[float],
    gti_pv: list[float],
) -> tuple[str, str]:
    """Build Plotly figures for the weather forecast dashboard.

    Args:
        labels:     HH:MM time labels.
        temps:      Outdoor temperature [C] per step.
        gti_window: South-facing window GTI [W/m²] per step.
        gti_pv:     PV panel GTI [W/m²] per step.

    Returns:
        Tuple (temperature_fig_json, solar_fig_json).
    """
    # Temperature figure
    t_fig = go.Figure()
    t_fig.add_trace(
        go.Scatter(
            x=labels,
            y=temps,
            name="T<sub>buiten</sub> [°C]",
            mode="lines+markers",
            line=dict(color="#1e6bbf", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(30,107,191,0.08)",
        )
    )
    t_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(
            title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=True, zerolinecolor="#ccc"
        ),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )

    # Solar / GTI figure
    s_fig = go.Figure()
    s_fig.add_trace(
        go.Scatter(
            x=labels,
            y=gti_window,
            name="GTI ramen [W/m²]",
            mode="lines",
            line=dict(color="#f39c12", width=2),
            fill="tozeroy",
            fillcolor="rgba(243,156,18,0.15)",
        )
    )
    if any(v > 0 for v in gti_pv):
        s_fig.add_trace(
            go.Scatter(
                x=labels,
                y=gti_pv,
                name="GTI PV-panelen [W/m²]",
                mode="lines",
                line=dict(color="#f1c40f", width=2, dash="dot"),
                fill="tozeroy",
                fillcolor="rgba(241,196,15,0.10)",
            )
        )
    s_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Instraling [W/m²]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return t_fig.to_json(), s_fig.to_json()


@app.get("/api/forecast", response_model=ForecastResponse)
async def get_forecast(
    latitude: float = Query(52.37, ge=-90.0, le=90.0, description="Site latitude [°N]"),
    longitude: float = Query(4.90, ge=-180.0, le=180.0, description="Site longitude [°E]"),
    horizon_hours: int = Query(48, ge=4, le=168, description="Forecast horizon [h]"),
    dt_hours: float = Query(1.0, ge=0.25, le=2.0, description="Time step [h]"),
    window_tilt: float = Query(
        90.0, ge=0.0, le=90.0, description="Window surface tilt [°]; 90 = vertical wall"
    ),
    window_azimuth: float = Query(
        0.0, ge=-180.0, le=180.0, description="Window azimuth [°]; 0 = South"
    ),
    pv_tilt: float | None = Query(
        None, ge=0.0, le=90.0, description="PV panel tilt [°]; None = no PV forecast"
    ),
    pv_azimuth: float = Query(
        0.0, ge=-180.0, le=180.0, description="PV panel azimuth [°]; 0 = South"
    ),
) -> ForecastResponse:
    """Fetch an OpenMeteo weather forecast and return structured JSON for the dashboard.

    Makes one or two HTTP calls to the Open-Meteo free API (no API key required).
    Returns temperature and solar irradiance arrays plus Plotly chart JSON.

    Args:
        latitude:       Site latitude [°N].
        longitude:      Site longitude [°E].
        horizon_hours:  Total forecast window [h].
        dt_hours:       Time step [h].
        window_tilt:    South-facing window tilt [°].
        window_azimuth: Window azimuth [°] (0 = South).
        pv_tilt:        PV panel tilt [°]; ``None`` disables the PV GTI fetch.
        pv_azimuth:     PV panel azimuth [°].

    Returns:
        ``ForecastResponse`` with arrays and Plotly chart JSON.

    Raises:
        HTTPException 502: When the Open-Meteo API call fails.
    """
    client = OpenMeteoClient(
        latitude=latitude,
        longitude=longitude,
        tilt=window_tilt,
        azimuth=window_azimuth,
        pv_tilt=pv_tilt,
        pv_azimuth=pv_azimuth,
    )
    try:
        forecast = client.get_forecast(
            horizon_hours=horizon_hours,
            dt_hours=dt_hours,
        )
    except (ConnectionError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    n = forecast.horizon_steps
    # Build HH:MM labels from valid_from UTC datetime
    base = forecast.valid_from
    labels = [
        (base + __import__("datetime").timedelta(hours=k * dt_hours)).strftime("%d-%m %H:%M")
        for k in range(n)
    ]
    temps = forecast.outdoor_temperature_c.tolist()
    gti_w = forecast.gti_w_per_m2.tolist()
    gti_pv = (
        forecast.gti_pv_w_per_m2.tolist() if forecast.gti_pv_w_per_m2 is not None else [0.0] * n
    )

    temp_fig_json, solar_fig_json = _build_forecast_figures(labels, temps, gti_w, gti_pv)

    return ForecastResponse(
        horizon_steps=n,
        dt_hours=dt_hours,
        valid_from=forecast.valid_from.isoformat(),
        labels=labels,
        outdoor_temperature_c=temps,
        gti_w_per_m2=gti_w,
        gti_pv_w_per_m2=gti_pv,
        temperature_forecast_fig=temp_fig_json,
        solar_forecast_fig=solar_fig_json,
    )


#: Environment variable that the addon sets so the API knows where the DB is.
#: Falls back to a local SQLite file in the current working directory.
#: Both constant and default are defined in ``database.py`` — do not repeat here.


def _get_repository() -> TelemetryRepository:
    """Construct a :class:`TelemetryRepository` from the active database URL.

    The URL is resolved by :meth:`~home_optimizer.database.Database.from_env`,
    which reads ``DATABASE_URL`` and falls back to the local SQLite default.

    ``Database.repository()`` calls ``create_schema()`` to ensure tables exist
    on a fresh database,
    so a missing table never causes a 502 — only an empty result (404).

    Returns
    -------
    TelemetryRepository
        Ready-to-use repository pointing at the configured database.
    """
    return Database.from_env().repository()


@app.get("/api/forecast/latest", response_model=ForecastResponse)
async def get_latest_forecast() -> ForecastResponse:
    """Return the most recently persisted Open-Meteo forecast from the database.

    Reads the latest batch from ``forecast_snapshots`` (highest
    ``fetched_at_utc``), builds Plotly chart JSON, and returns a
    :class:`ForecastResponse` — the same shape as ``GET /api/forecast``.

    The database is located via the ``DATABASE_URL`` environment variable
    (set by the HA addon supervisor).  Local default:
    ``sqlite:///database.sqlite3`` relative to the current working directory.

    Returns
    -------
    ForecastResponse
        Latest forecast from the database.

    Raises
    ------
    HTTPException 404
        When the ``forecast_snapshots`` table is empty (no forecast has been
        persisted yet — run the addon or ``ForecastPersister.persist_once()``
        first).
    HTTPException 502
        When the database cannot be reached.
    """
    try:
        repo = _get_repository()
        rows = repo.get_latest_forecast_batch()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Database error: {exc}") from exc

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geen forecast gevonden in de database. "
                "Start de addon of roep ForecastPersister.persist_once() aan "
                "om de eerste forecast op te slaan."
            ),
        )

    fetched_at = rows[0].fetched_at_utc
    dt_hours = float(rows[0].dt_hours)
    n = len(rows)

    # Build time labels from valid_at_utc stored in each row
    labels = [row.valid_at_utc.strftime("%d-%m %H:%M") for row in rows]
    temps = [float(row.t_out_c) for row in rows]
    gti_w = [float(row.gti_w_per_m2) for row in rows]
    gti_pv = [float(row.gti_pv_w_per_m2) for row in rows]

    temp_fig_json, solar_fig_json = _build_forecast_figures(labels, temps, gti_w, gti_pv)

    return ForecastResponse(
        horizon_steps=n,
        dt_hours=dt_hours,
        valid_from=fetched_at.isoformat(),
        labels=labels,
        outdoor_temperature_c=temps,
        gti_w_per_m2=gti_w,
        gti_pv_w_per_m2=gti_pv,
        temperature_forecast_fig=temp_fig_json,
        solar_forecast_fig=solar_fig_json,
    )


@app.post("/api/simulate", response_model=OptimizeResponse)
async def optimize(req: RunRequest) -> OptimizeResponse:  # noqa: C901
    """Run one MPC optimisation step and return charts + numerical summaries.

    Processing pipeline:

    1. Delegate numerical solve to :meth:`~home_optimizer.optimizer.Optimizer.solve`.
    2. Compute cost-attribution summaries.
    3. Serialise all Plotly charts to JSON for the browser.

    Args:
        req: Validated request with all physical and MPC parameters.

    Returns:
        ``OptimizeResponse`` with numerical results and Plotly chart JSON.

    Raises:
        HTTPException 422: If any parameter is physically invalid.
    """
    try:
        result = Optimizer().solve(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    N = req.horizon_hours
    start_hour = result.start_hour
    P_UFH = result.p_ufh_kw
    P_dhw = result.p_dhw_kw
    cop_ufh_arr = result.cop_ufh_arr
    cop_dhw_arr = result.cop_dhw_arr
    pv_kw = result.pv_kw
    solution = result.solution
    states = solution.predicted_states_c
    prices = result.ufh_forecast.price_eur_per_kwh
    dt = req.dt_hours

    # ── Cost / energy attribution (electrical basis, §14.1) ──────────────
    P_UFH_elec = P_UFH / cop_ufh_arr
    P_dhw_elec = P_dhw / cop_dhw_arr
    P_hp_total_elec = P_UFH_elec + P_dhw_elec
    P_hp_total_therm = P_UFH + P_dhw
    P_import = np.maximum(P_hp_total_elec - pv_kw, 0.0)
    total_energy = float(np.sum(P_hp_total_therm) * dt)
    total_cost_steps = P_import * prices * dt
    total_cost = float(np.sum(total_cost_steps))
    pv_total = float(np.sum(pv_kw) * dt)
    net_grid = float(np.sum(P_import) * dt)
    ufh_energy = result.ufh_energy_kwh
    dhw_energy = result.dhw_energy_kwh
    solver_status = solution.solver_status
    objective = solution.objective_value
    max_comfort_viol = solution.max_ufh_comfort_violation_c
    max_dhw_viol = solution.max_dhw_comfort_violation_c
    max_leg_viol = solution.max_legionella_violation_c
    ufh_cost_weights = np.divide(
        P_UFH_elec,
        P_hp_total_elec,
        out=np.zeros_like(P_UFH_elec),
        where=P_hp_total_elec > 0.0,
    )
    dhw_cost_weights = np.divide(
        P_dhw_elec,
        P_hp_total_elec,
        out=np.zeros_like(P_dhw_elec),
        where=P_hp_total_elec > 0.0,
    )
    ufh_grid_cost = float(np.sum(total_cost_steps * ufh_cost_weights))
    dhw_grid_cost = float(np.sum(total_cost_steps * dhw_cost_weights))

    # ── 7. Build charts ──────────────────────────────────────────────────
    labels_states = _time_labels(start_hour, N + 1)
    labels_ctrl = _time_labels(start_hour, N)
    T_r = states[:, 0]

    temp_fig = _temperature_figure(
        labels_states,
        T_r,
        req.T_ref,
        req.T_min,
        req.T_max,
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
