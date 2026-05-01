from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import plotly
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse

from home_optimizer.app import AppSettings
from home_optimizer.web.cache import NO_CACHE_HEADERS
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.pages import build_dashboard_view_model, render_template
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import DashboardChartsResponse
from home_optimizer.web.services import DashboardChartsService

PLOTLY_JS_PATH = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
ChartDateQuery = Annotated[date, Query(alias="date")]
ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]


def create_dashboard_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        view_model = build_dashboard_view_model(
            settings,
            title="Home Optimizer",
        )
        return HTMLResponse(render_template("dashboard.html", view_model))

    @router.get("/plotly.js", response_class=FileResponse)
    def plotly_js() -> FileResponse:
        return FileResponse(
            PLOTLY_JS_PATH,
            media_type="application/javascript",
            headers=NO_CACHE_HEADERS,
        )

    @router.get("/api/dashboard/charts", response_model=DashboardChartsResponse)
    def get_dashboard_charts(
        chart_date: ChartDateQuery,
        container: ContainerDependency,
    ) -> DashboardChartsResponse:
        return DashboardChartsService(container.time_series_read_repository).get_day_charts(chart_date)

    return router
