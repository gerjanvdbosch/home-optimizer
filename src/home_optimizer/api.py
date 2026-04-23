"""FastAPI web interface for the Home Optimizer.

This module exposes a class-based API surface where route handlers and helper
logic live on ``HomeOptimizerAPI``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from pydantic import BaseModel

from .application import optimizer as optimizer_module
from .application.optimizer import (
    MPCStepResult,
    Optimizer,
    RunRequest,
    build_repository_forecast_overrides,
    merge_run_request_updates,
    sanitize_calibration_overrides,
)
from .telemetry import TelemetryRepository
from .types.calibration import CalibrationSnapshotPayload


class OptimizeResponse(BaseModel):
    """Structured optimisation response for dashboard and API clients."""

    status: str
    objective: float
    hp_total_energy_kwh: float
    total_cost_eur: float
    ufh_total_energy_kwh: float
    dhw_total_energy_kwh: float = 0.0
    ufh_grid_cost_eur: float
    dhw_grid_cost_eur: float = 0.0
    first_ufh_power_kw: float
    first_dhw_power_kw: float = 0.0
    first_total_hp_power_kw: float
    max_ufh_comfort_violation_c: float
    pv_total_kwh: float = 0.0
    net_grid_energy_kwh: float = 0.0
    pv_enabled: bool = False
    control_labels: list[str]
    pv_forecast_kw: list[float]
    dhw_enabled: bool = False
    max_dhw_comfort_violation_c: float = 0.0
    max_legionella_violation_c: float = 0.0
    cop_ufh_profile: list[float] = []
    cop_dhw_profile: list[float] = []
    temperature_fig: str
    power_fig: str
    cop_fig: str = ""
    pv_forecast_fig: str = ""
    dhw_fig: str = ""
    dhw_top_profile_c: list[float] = []
    dhw_bottom_profile_c: list[float] = []
    dhw_target_profile_c: list[float] = []
    dhw_tap_profile_m3_per_h: list[float] = []


class ForecastResponse(BaseModel):
    """OpenMeteo forecast response for the operational dashboard."""

    horizon_steps: int
    dt_hours: float
    valid_from: str
    labels: list[str]
    outdoor_temperature_c: list[float]
    gti_w_per_m2: list[float]
    gti_pv_w_per_m2: list[float]
    temperature_forecast_fig: str
    solar_forecast_fig: str


class HomeOptimizerAPI:
    """Class-based HTTP layer for Home Optimizer."""

    def __init__(self) -> None:
        self._base_request: RunRequest | None = None
        self.app = FastAPI(
            title="Home Optimizer",
            description=(
                "Gecombineerd thermisch model (UFH + DHW) met MPC-optimalisatie, "
                "Carnot COP-model, Kalman-filter toestandschatting en PV self-consumption."
            ),
            version="0.9",
        )
        self._template_dashboard = Path(__file__).parent / "templates" / "dashboard.html"
        self._template_simulator = Path(__file__).parent / "templates" / "simulator.html"
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_api_route(
            "/",
            self.index,
            methods=["GET"],
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        self.app.add_api_route(
            "/simulator",
            self.simulator,
            methods=["GET"],
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        self.app.add_api_route("/api/defaults", self.defaults, methods=["GET"])
        self.app.add_api_route(
            "/api/forecast/latest",
            self.get_latest_forecast,
            methods=["GET"],
            response_model=ForecastResponse,
        )
        self.app.add_api_route(
            "/api/simulate",
            self.simulate,
            methods=["POST"],
            response_model=OptimizeResponse,
        )
        self.app.add_api_route(
            "/api/optimizer/latest",
            self.latest_optimizer_result,
            methods=["GET"],
            response_model=OptimizeResponse,
        )
        self.app.add_api_route(
            "/api/calibration/latest",
            self.latest_calibration_snapshot,
            methods=["GET"],
            response_model=CalibrationSnapshotPayload,
        )

    async def index(self) -> HTMLResponse:
        """Serve operational dashboard HTML."""
        return HTMLResponse(self._template_dashboard.read_text(encoding="utf-8"))

    async def simulator(self) -> HTMLResponse:
        """Serve simulator HTML."""
        return HTMLResponse(self._template_simulator.read_text(encoding="utf-8"))

    def set_base_request(self, base_request: RunRequest | None) -> None:
        """Store the canonical runtime base request used by API defaults/simulations.

        Args:
            base_request: Fully validated runtime request assembled by the addon or
                local runner. ``None`` restores the plain ``RunRequest`` defaults.
        """

        self._base_request = base_request

    def _runtime_base_request(self) -> RunRequest:
        """Return the canonical runtime base request for interactive API calls."""

        if self._base_request is not None:
            return self._base_request
        return RunRequest.model_validate({})

    async def defaults(self) -> RunRequest:
        """Return calibrated ``RunRequest`` defaults when a snapshot is available."""
        defaults = self._runtime_base_request()
        try:
            snapshot = self._get_repository().get_latest_calibration_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Database error: {exc}") from exc
        if snapshot is None or not snapshot.has_effective_parameters:
            return defaults
        safe_overrides, _ = sanitize_calibration_overrides(defaults, snapshot.effective_parameters)
        return merge_run_request_updates(defaults, safe_overrides.as_run_request_updates())


    async def get_latest_forecast(self) -> ForecastResponse:
        """Return the most recently persisted weather forecast from DB."""
        try:
            rows = self._get_repository().get_latest_forecast_batch()
        except Exception as exc:  # noqa: BLE001
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

        dt_hours = float(rows[0].dt_hours)
        labels = [row.valid_at_utc.strftime("%d-%m %H:%M") for row in rows]
        temps = [float(row.t_out_c) for row in rows]
        gti_w = [float(row.gti_w_per_m2) for row in rows]
        gti_pv = [float(row.gti_pv_w_per_m2) for row in rows]
        temp_fig_json, solar_fig_json = self._build_forecast_figures(labels, temps, gti_w, gti_pv)

        return ForecastResponse(
            horizon_steps=len(rows),
            dt_hours=dt_hours,
            valid_from=rows[0].fetched_at_utc.isoformat(),
            labels=labels,
            outdoor_temperature_c=temps,
            gti_w_per_m2=gti_w,
            gti_pv_w_per_m2=gti_pv,
            temperature_forecast_fig=temp_fig_json,
            solar_forecast_fig=solar_fig_json,
        )

    async def simulate(self, req: RunRequest) -> OptimizeResponse:
        """Run one MPC optimisation and return charts + summaries.

        Injects the latest Open-Meteo forecast from the database into the
        request before solving (for any array not already supplied by the
        caller).  Raises 502 on DB error, 422 when no forecast rows exist.
        """
        repository = self._get_repository()
        effective_request = merge_run_request_updates(
            self._runtime_base_request(),
            req.model_dump(mode="python", exclude_unset=True),
        )

        try:
            calibration_overrides = optimizer_module.build_safe_calibration_overrides(effective_request, repository)
            rows, forecast_overrides = build_repository_forecast_overrides(
                effective_request,
                repository,
                existing=calibration_overrides,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Kon forecast niet uit de database lezen: {exc}",
            ) from exc

        if effective_request.gti_window_forecast is None or effective_request.t_out_forecast is None:
            if not rows:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Geen forecast gevonden in de database.  "
                        "Zorg dat ForecastPersister minstens één keer heeft gedraaid "
                        "voordat je /api/simulate aanroept."
                    ),
                )
        elif effective_request.shutter_forecast is not None or not rows:
            forecast_overrides = {}

        effective_overrides = {**calibration_overrides, **forecast_overrides}
        if effective_overrides:
            effective_request = merge_run_request_updates(effective_request, effective_overrides)

        try:
            result = Optimizer().solve(effective_request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return self._build_optimize_response(req=effective_request, result=result)



    async def latest_optimizer_result(self) -> OptimizeResponse:
        """Return latest successful periodic Optimizer result."""
        snapshot = Optimizer.get_latest_scheduled_snapshot()
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail="Nog geen geplande MPC-uitkomst beschikbaar. Wacht op de eerste scheduler-run.",
            )
        return self._build_optimize_response(req=snapshot.request, result=snapshot.result)

    async def latest_calibration_snapshot(self) -> CalibrationSnapshotPayload:
        """Return the latest persisted automatic calibration snapshot."""
        try:
            snapshot = self._get_repository().get_latest_calibration_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Database error: {exc}") from exc
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail="Nog geen calibration snapshot beschikbaar. Wacht op de eerste automatische calibratie-run.",
            )
        return snapshot

    @staticmethod
    def _get_repository() -> TelemetryRepository:
        """Build telemetry repository from active ``DATABASE_URL``."""
        return TelemetryRepository.from_env()

    @staticmethod
    def _time_labels(start_hour: int, n_points: int) -> list[str]:
        """Generate HH:MM labels starting from ``start_hour``."""
        base = datetime.now(tz=timezone.utc).replace(hour=start_hour, minute=0, second=0, microsecond=0)
        return [(base + timedelta(hours=k)).strftime("%H:%M") for k in range(n_points)]

    @staticmethod
    def _build_forecast_figures(
        labels: list[str],
        temps: list[float],
        gti_window: list[float],
        gti_pv: list[float],
    ) -> tuple[str, str]:
        """Build weather forecast figures (temperature + irradiance)."""
        # Use numeric x + ticktext for consistency across all dashboard charts
        t_fig = go.Figure()
        n = len(labels)
        x_vals = list(range(n))
        t_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=temps,
                name="T<sub>buiten</sub> [degC]",
                mode="lines+markers",
                line=dict(color="#1e6bbf", width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(30,107,191,0.08)",
            )
        )
        t_fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        t_fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Temperatuur [degC]", gridcolor="rgba(0,0,0,0.04)", zeroline=True, zerolinecolor="rgba(0,0,0,0.06)"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )

        s_fig = go.Figure()
        s_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=gti_window,
                name="GTI ramen [W/m2]",
                mode="lines",
                line=dict(color="#f39c12", width=2),
                fill="tozeroy",
                fillcolor="rgba(243,156,18,0.15)",
            )
        )
        s_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=gti_pv,
                name="GTI PV-panelen [W/m2]",
                mode="lines",
                line=dict(color="#f1c40f", width=2, dash="dot"),
                fill="tozeroy",
                fillcolor="rgba(241,196,15,0.10)",
            )
        )
        s_fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        s_fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Instraling [W/m2]", gridcolor="rgba(0,0,0,0.04)", zeroline=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return t_fig.to_json(), s_fig.to_json()

    @staticmethod
    def _temperature_figure(
        labels: list[str],
        t_r: np.ndarray,
        t_ref: float,
        t_min: float,
        t_max: float,
        t_out: np.ndarray,
    ) -> str:
        """Build UFH room-temperature figure.

        Args:
            labels: Time-axis labels (length N+1, one per state step).
            t_r:    Predicted room-air temperature [°C], length N+1.
            t_ref:  Comfort setpoint reference line [°C].
            t_min:  Lower comfort band [°C].
            t_max:  Upper comfort band [°C].
            t_out:  Outdoor temperature forecast array [°C], length N+1.
                    Plotted as a dotted reference trace.
        """
        # Use numeric x-values to avoid categorical string-collapsing when labels
        # (HH:MM) repeat across days. The visible tick labels remain the provided
        # "labels" via ticktext so the axis still shows human-friendly times.
        fig = go.Figure()
        n = len(labels)
        x_vals = list(range(n))

        # Comfort band polygon — use numeric x to avoid Plotly category merging
        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=[t_max] * n + [t_min] * n,
                fill="toself",
                fillcolor="rgba(100,149,237,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Comfortband",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=t_out,
                name="T<sub>buiten</sub> (forecast)",
                mode="lines",
                line=dict(color="#999", width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[t_ref] * n,
                name="T<sub>ref</sub>",
                mode="lines",
                line=dict(color="#2ca02c", width=1.5, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=t_r,
                name="T<sub>r</sub> (kamer)",
                mode="lines+markers",
                line=dict(color="#1e6bbf", width=2.5),
                marker=dict(size=5),
            )
        )

        # Show human-readable time labels while keeping x numeric for plotting
        fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        # Reduce visual noise: use a transparent plot background so the card
        # background carries the visual weight, hide x-gridlines and make y-grid
        # lines very subtle. This focuses attention on the traces themselves.
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Temperatuur [degC]", gridcolor="rgba(0,0,0,0.04)", zeroline=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return fig.to_json()

    @staticmethod
    def _dhw_figure(
        labels: list[str],
        t_top: np.ndarray,
        t_bot: np.ndarray,
        t_dhw_min: float,
        t_dhw_target: np.ndarray | None,
        v_tap_m3_per_h: np.ndarray | None,
    ) -> str:
        """Build DHW tank-temperature figure."""
        # Use numeric x-values and explicit tick labels to prevent duplicate
        # categorical labels (same HH:MM across multiple days) from collapsing
        # separate time points onto a single x-category.
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        n = len(labels)
        x_vals = list(range(n))

        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=[t_dhw_min] * n + [20.0] * n,
                fill="toself",
                fillcolor="rgba(255,165,0,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Comfort min",
                hoverinfo="skip",
            )
        )
        if t_dhw_target is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=t_dhw_target,
                    name="T<sub>target</sub>",
                    mode="lines",
                    line=dict(color="#8e44ad", width=2, dash="dash"),
                )
            )
        if v_tap_m3_per_h is not None:
            fig.add_trace(
                go.Bar(
                    x=x_vals[:-1],
                    y=v_tap_m3_per_h,
                    name="V̇<sub>tap</sub>",
                    marker_color="rgba(52,152,219,0.30)",
                ),
                secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=t_top,
                name="T<sub>top</sub>",
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2.5),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=t_bot,
                name="T<sub>bot</sub>",
                mode="lines+markers",
                line=dict(color="#f39c12", width=2),
                marker=dict(size=4),
            )
        )

        # Keep human-readable time labels
        fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        # Subtle grid and transparent plotting area to reduce background
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Temperatuur [degC]", gridcolor="rgba(0,0,0,0.04)", zeroline=False),
            yaxis2=dict(title="Tapflow [m³/h]", rangemode="tozero", showgrid=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return fig.to_json()

    @staticmethod
    def _power_figure(
        labels: list[str],
        p_ufh: np.ndarray,
        p_dhw: np.ndarray,
        pv_kw: np.ndarray,
        prices: np.ndarray,
        p_max: float,
        baseload_kw: np.ndarray | None = None,
    ) -> str:
        """Build power + PV + price figure.

        Adds an optional baseload (electrical household demand) line to the
        primary y-axis so the user can compare heat-pump power vs household
        consumption.
        """
        # numeric x + ticktext for consistency; baseload (electrical) can be
        # plotted as a line on the primary axis to show household demand.
        n = len(labels)
        x_vals = list(range(n))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=p_ufh,
                name="P<sub>UFH</sub> [kW therm]",
                marker_color=[f"rgba(30,107,191,{0.5 + 0.5 * float(v) / max(p_max, 0.01)})" for v in p_ufh],
            ),
            secondary_y=False,
        )
        if np.any(p_dhw > 0):
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=p_dhw,
                    name="P<sub>DHW</sub> [kW therm]",
                    marker_color="rgba(231,76,60,0.65)",
                ),
                secondary_y=False,
            )
        if np.any(pv_kw > 0):
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=pv_kw,
                    name="P<sub>PV</sub> [kW]",
                    mode="lines",
                    line=dict(color="#f1c40f", width=2, dash="dot"),
                ),
                secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=prices,
                name="Prijs [EUR/kWh]",
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )
        # Baseload: optional electrical household demand forecast
        if baseload_kw is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=baseload_kw,
                    name="Baseload [kW elec]",
                    mode="lines",
                    line=dict(color="#7f8c8d", width=2, dash="dash"),
                ),
                secondary_y=False,
            )
        fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            hovermode="x unified",
            barmode="group",
        )
        fig.update_yaxes(
            title_text="Thermisch vermogen [kW]",
            secondary_y=False,
            range=[0, p_max * 1.1],
            gridcolor="rgba(0,0,0,0.04)",
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="Prijs [EUR/kWh]",
            secondary_y=True,
            range=[0, max(prices) * 1.5],
            gridcolor=None,
            zeroline=False,
            showgrid=False,
        )
        # Make x-grid less prominent / hidden so price and power bars stand out
        fig.update_xaxes(showgrid=False)
        return fig.to_json()

    @staticmethod
    def _cop_figure(
        labels: list[str],
        cop_ufh: np.ndarray,
        cop_dhw: np.ndarray | None,
        cop_min: float,
    ) -> str:
        """Build COP profile figure."""
        # numeric x + ticktext for consistency
        fig = go.Figure()
        n = len(labels)
        x_vals = list(range(n))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
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
                    x=x_vals,
                    y=cop_dhw,
                    name="COP<sub>DHW</sub> (vaste aanvoertemp)",
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=2),
                    marker=dict(size=5),
                    fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.06)",
                )
            )
        fig.add_hline(
            y=cop_min,
            line=dict(color="#aaa", width=1, dash="dot"),
            annotation_text=f"COP_min = {cop_min}",
            annotation_position="top left",
            annotation_font_size=10,
        )
        fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="COP [-]", gridcolor="rgba(0,0,0,0.04)", zeroline=False, rangemode="tozero"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return fig.to_json()

    @staticmethod
    def _pv_forecast_figure(labels: list[str], pv_kw: np.ndarray) -> str:
        """Build PV forecast figure."""
        fig = go.Figure()
        n = len(labels)
        x_vals = list(range(n))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pv_kw,
                name="P<sub>PV</sub> forecast [kW]",
                mode="lines+markers",
                line=dict(color="#f1c40f", width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(241,196,15,0.18)",
            )
        )
        fig.update_xaxes(tickmode="array", tickvals=x_vals, ticktext=labels)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="PV forecast [kW]", gridcolor="rgba(0,0,0,0.04)", zeroline=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return fig.to_json()

    def _build_optimize_response(self, req: RunRequest, result: MPCStepResult) -> OptimizeResponse:
        """Build one ``OptimizeResponse`` from an optimizer run."""
        n = req.horizon_hours
        start_hour = result.start_hour
        p_ufh = result.p_ufh_kw
        p_dhw = result.p_dhw_kw
        cop_ufh_arr = result.cop_ufh_arr
        cop_dhw_arr = result.cop_dhw_arr
        pv_kw = result.pv_kw
        solution = result.solution
        states = solution.predicted_states_c
        prices = result.ufh_forecast.price_eur_per_kwh
        dt = req.dt_hours

        p_ufh_elec = p_ufh / cop_ufh_arr
        p_dhw_elec = p_dhw / cop_dhw_arr
        p_hp_total_elec = p_ufh_elec + p_dhw_elec
        p_hp_total_therm = p_ufh + p_dhw
        p_import = np.maximum(p_hp_total_elec - pv_kw, 0.0)
        total_energy = float(np.sum(p_hp_total_therm) * dt)
        total_cost_steps = p_import * prices * dt
        total_cost = float(np.sum(total_cost_steps))
        pv_total = float(np.sum(pv_kw) * dt)
        net_grid = float(np.sum(p_import) * dt)
        ufh_energy = result.ufh_energy_kwh
        dhw_energy = result.dhw_energy_kwh
        solver_status = solution.solver_status
        objective = solution.objective_value
        max_comfort_viol = solution.max_ufh_comfort_violation_c
        max_dhw_viol = solution.max_dhw_comfort_violation_c
        max_leg_viol = solution.max_legionella_violation_c
        ufh_cost_weights = np.divide(
            p_ufh_elec,
            p_hp_total_elec,
            out=np.zeros_like(p_ufh_elec),
            where=p_hp_total_elec > 0.0,
        )
        dhw_cost_weights = np.divide(
            p_dhw_elec,
            p_hp_total_elec,
            out=np.zeros_like(p_dhw_elec),
            where=p_hp_total_elec > 0.0,
        )
        ufh_grid_cost = float(np.sum(total_cost_steps * ufh_cost_weights))
        dhw_grid_cost = float(np.sum(total_cost_steps * dhw_cost_weights))

        labels_states = self._time_labels(start_hour, n + 1)
        labels_ctrl = self._time_labels(start_hour, n)
        t_r = states[:, 0]

        # Outdoor temperature: use the actual forecast array used by the solver.
        # Pad to N+1 by repeating the last value so it aligns with state labels.
        t_out_arr = result.ufh_forecast.outdoor_temperature_c
        t_out_states = np.append(t_out_arr, t_out_arr[-1])

        temp_fig = self._temperature_figure(
            labels_states,
            t_r,
            req.T_ref,
            req.T_min,
            req.T_max,
            t_out_states,
        )
        power_fig = self._power_figure(labels_ctrl, p_ufh, p_dhw, pv_kw, prices, req.P_max)
        # Baseload: optional household electrical demand forecast supplied in the
        # RunRequest (may have been injected by the ForecastService). Materialize
        # to length N and fall back to zeros when absent.
        if req.baseload_forecast is None:
            baseload_kw = np.zeros(len(labels_ctrl), dtype=float)
        else:
            arr = np.asarray(req.baseload_forecast, dtype=float)
            if arr.size >= len(labels_ctrl):
                baseload_kw = arr[: len(labels_ctrl)].copy()
            else:
                # pad by repeating last known value to reach horizon length
                pad_len = len(labels_ctrl) - arr.size
                pad_val = arr[-1] if arr.size > 0 else 0.0
                baseload_kw = np.concatenate([arr, np.full(pad_len, pad_val, dtype=float)])
        power_fig = self._power_figure(labels_ctrl, p_ufh, p_dhw, pv_kw, prices, req.P_max, baseload_kw=baseload_kw)
        cop_fig = self._cop_figure(
            labels_ctrl,
            cop_ufh=cop_ufh_arr,
            cop_dhw=cop_dhw_arr if req.dhw_enabled else None,
            cop_min=req.cop_min,
        )
        pv_forecast_fig = self._pv_forecast_figure(labels_ctrl, pv_kw)
        dhw_fig = ""
        dhw_top_profile: list[float] = []
        dhw_bottom_profile: list[float] = []
        dhw_target_profile: list[float] = []
        dhw_tap_profile: list[float] = []
        if req.dhw_enabled:
            t_top = states[:, 2]
            t_bot = states[:, 3]
            dhw_top_profile = t_top.tolist()
            dhw_bottom_profile = t_bot.tolist()
            dhw_target_arr = (
                result.dhw_forecast.target_top_c
                if result.dhw_forecast is not None and result.dhw_forecast.target_top_c is not None
                else np.full(n, req.dhw_T_target, dtype=float)
            )
            dhw_target_profile = np.append(dhw_target_arr, dhw_target_arr[-1]).tolist()
            dhw_tap_arr = (
                result.dhw_forecast.v_tap_m3_per_h
                if result.dhw_forecast is not None
                else np.zeros(n, dtype=float)
            )
            dhw_tap_profile = dhw_tap_arr.tolist()
            dhw_fig = self._dhw_figure(
                labels_states,
                t_top,
                t_bot,
                req.dhw_T_min,
                np.append(dhw_target_arr, dhw_target_arr[-1]),
                dhw_tap_arr,
            )

        return OptimizeResponse(
            status=solver_status,
            objective=objective,
            hp_total_energy_kwh=total_energy,
            total_cost_eur=total_cost,
            ufh_total_energy_kwh=ufh_energy,
            dhw_total_energy_kwh=dhw_energy,
            ufh_grid_cost_eur=ufh_grid_cost,
            dhw_grid_cost_eur=dhw_grid_cost,
            first_ufh_power_kw=float(p_ufh[0]),
            first_dhw_power_kw=float(p_dhw[0]),
            first_total_hp_power_kw=float(p_hp_total_therm[0]),
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
            dhw_top_profile_c=dhw_top_profile,
            dhw_bottom_profile_c=dhw_bottom_profile,
            dhw_target_profile_c=dhw_target_profile,
            dhw_tap_profile_m3_per_h=dhw_tap_profile,
        )


api_service = HomeOptimizerAPI()
app = api_service.app
