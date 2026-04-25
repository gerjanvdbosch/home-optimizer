from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from home_optimizer.app.container_factories import build_home_assistant_container
from home_optimizer.app.history_import_jobs import HistoryImportJob, HistoryImportJobRunner
from home_optimizer.app.history_import_requests import build_history_import_request
from home_optimizer.app.settings import AppSettings
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult
from home_optimizer.web.pages import render_dashboard
from home_optimizer.web.schemas import (
    DashboardViewModel,
    HistoryImportJobResponse,
    HistoryImportRunResponse,
)

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ClosableGateway(Protocol):
    def close(self) -> None: ...


class HistoryImportRunner(Protocol):
    def import_many(self, request: HistoryImportRequest) -> HistoryImportResult: ...


class LiveCollectionSchedulerRunner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def live_collection_scheduler(self) -> LiveCollectionSchedulerRunner: ...


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
        container.live_collection_scheduler.start()
        try:
            yield
        finally:
            container.live_collection_scheduler.stop()
            app.state.history_import_jobs.shutdown()
            container.home_assistant.close()

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
            import_enabled=settings.history_import_enabled,
            import_window_days=settings.history_import_max_days_back,
            chunk_days=settings.history_import_chunk_days,
            sensor_count=sensor_count,
            database_path=settings.database_path,
            api_port=settings.api_port,
        )
        return HTMLResponse(render_dashboard(view_model))

    @app.post("/api/history-import", response_model=HistoryImportRunResponse)
    def run_history_import() -> HistoryImportRunResponse:
        if not settings.history_import_enabled:
            raise HTTPException(status_code=409, detail="History import is uitgeschakeld.")

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
