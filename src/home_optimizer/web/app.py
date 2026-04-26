from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, Callable, Protocol

import plotly
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from home_optimizer.app.container_factories import build_home_assistant_container
from home_optimizer.app.history_import_jobs import HistoryImportJob, HistoryImportJobRunner
from home_optimizer.app.history_import_requests import build_history_import_request
from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.charts import ChartSeries, ChartTextSeries
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult
from home_optimizer.web.pages import render_dashboard
from home_optimizer.web.schemas import (
    ChartPointResponse,
    ChartSeriesResponse,
    ChartTextPointResponse,
    ChartTextSeriesResponse,
    DashboardChartsResponse,
    DashboardViewModel,
    HistoryImportJobResponse,
    HistoryImportRunResponse,
)

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
PLOTLY_JS_PATH = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
ChartDateQuery = Annotated[date, Query(alias="date")]


class ClosableGateway(Protocol):
    def close(self) -> None: ...


class HistoryImportRunner(Protocol):
    def import_many(self, request: HistoryImportRequest) -> HistoryImportResult: ...


class DashboardDataReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartSeries]: ...

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartTextSeries]: ...


class TelemetrySchedulerRunner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def dashboard_repository(self) -> DashboardDataReader: ...

    @property
    def telemetry_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def forecast_scheduler(self) -> TelemetrySchedulerRunner: ...

    def close(self) -> None: ...


def _build_history_request(
    settings: AppSettings,
) -> tuple[HistoryImportRequest, int]:
    request = build_history_import_request(settings)
    return request, len(request.specs)


def _job_response(job: HistoryImportJob) -> HistoryImportJobResponse:
    return HistoryImportJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        imported_rows=job.imported_rows,
        total_rows=job.total_rows,
        sensor_count=job.sensor_count,
        error=job.error,
    )


def _series_response(series: ChartSeries) -> ChartSeriesResponse:
    return ChartSeriesResponse(
        name=series.name,
        unit=series.unit,
        points=[
            ChartPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def _text_series_response(series: ChartTextSeries) -> ChartTextSeriesResponse:
    return ChartTextSeriesResponse(
        name=series.name,
        points=[
            ChartTextPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def create_app(
    settings: AppSettings,
    container_factory: Callable[[AppSettings], WebAppContainer] = build_home_assistant_container,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = container_factory(settings)
        app.state.container = container
        app.state.history_import_jobs = HistoryImportJobRunner(
            settings=settings,
            importer=container.history_import_service,
        )
        container.telemetry_scheduler.start()
        container.forecast_scheduler.start()
        try:
            yield
        finally:
            container.forecast_scheduler.stop()
            container.telemetry_scheduler.stop()
            app.state.history_import_jobs.shutdown()
            container.close()

    app = FastAPI(
        title="Home Optimizer API",
        version="0.10",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_container() -> WebAppContainer:
        return app.state.container

    def get_history_import_jobs() -> HistoryImportJobRunner:
        return app.state.history_import_jobs

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        _, sensor_count = _build_history_request(settings)
        view_model = DashboardViewModel(
            title="Home Optimizer",
            import_window_days=settings.history_import_max_days_back,
            chunk_days=settings.history_import_chunk_days,
            sensor_count=sensor_count,
            database_path=settings.database_path,
            api_port=settings.api_port,
        )
        return HTMLResponse(render_dashboard(view_model))

    @app.get("/plotly.js", response_class=FileResponse)
    def plotly_js() -> FileResponse:
        return FileResponse(PLOTLY_JS_PATH, media_type="application/javascript")

    @app.get("/api/dashboard/charts", response_model=DashboardChartsResponse)
    def get_dashboard_charts(
        chart_date: ChartDateQuery,
    ) -> DashboardChartsResponse:
        start_time = datetime.combine(chart_date, time.min, tzinfo=timezone.utc)
        end_time = start_time + timedelta(days=1)
        series = get_container().dashboard_repository.read_series(
            names=[
                "room_temperature",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = get_container().dashboard_repository.read_text_series(
            names=["hp_mode"],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = {item.name: item for item in series}
        text_series_by_name = {item.name: item for item in text_series}

        return DashboardChartsResponse(
            date=chart_date.isoformat(),
            room_temperature=_series_response(series_by_name["room_temperature"]),
            dhw_temperatures=[
                _series_response(series_by_name["dhw_top_temperature"]),
                _series_response(series_by_name["dhw_bottom_temperature"]),
            ],
            heatpump_power=_series_response(series_by_name["hp_electric_power"]),
            heatpump_mode=_text_series_response(text_series_by_name["hp_mode"]),
        )

    @app.post("/api/history-import", response_model=HistoryImportRunResponse)
    def run_history_import() -> HistoryImportRunResponse:
        _, sensor_count = _build_history_request(settings)
        if sensor_count == 0:
            raise HTTPException(status_code=400, detail="Geen sensoren geconfigureerd voor import.")

        job = get_history_import_jobs().start()
        LOGGER.info("History import job started: %s", job.job_id)

        return HistoryImportRunResponse(
            job_id=job.job_id,
            status=job.status,
            sensor_count=sensor_count,
        )

    @app.get("/api/history-import/jobs/{job_id}", response_model=HistoryImportJobResponse)
    def get_history_import_job(job_id: str) -> HistoryImportJobResponse:
        job = get_history_import_jobs().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Import job niet gevonden.")

        return _job_response(job)

    return app
