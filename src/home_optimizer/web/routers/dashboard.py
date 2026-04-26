from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from string import Template
from typing import Annotated

import plotly
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse

from home_optimizer.app.history_import_requests import build_history_import_request
from home_optimizer.app.settings import AppSettings
from home_optimizer.web.cache import NO_CACHE_HEADERS
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import DashboardChartsResponse, DashboardViewModel
from home_optimizer.web.services.dashboard_charts import DashboardChartsService

PLOTLY_JS_PATH = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
ChartDateQuery = Annotated[date, Query(alias="date")]
ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]


def render_dashboard(view_model: DashboardViewModel) -> str:
    template = Template((TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8"))
    return template.substitute(
        title=escape(view_model.title),
        import_window_days=view_model.import_window_days,
        chunk_days=view_model.chunk_days,
        sensor_count=view_model.sensor_count,
        database_path=escape(view_model.database_path),
        api_port=view_model.api_port,
    )


def create_dashboard_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        request = build_history_import_request(settings)
        view_model = DashboardViewModel(
            title="Home Optimizer",
            import_window_days=settings.history_import_max_days_back,
            chunk_days=settings.history_import_chunk_days,
            sensor_count=len(request.specs),
            database_path=settings.database_path,
            api_port=settings.api_port,
        )
        return HTMLResponse(render_dashboard(view_model))

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
        return DashboardChartsService(container.dashboard_repository).get_day_charts(chart_date)

    return router
