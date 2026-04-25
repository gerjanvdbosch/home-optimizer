from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from home_optimizer.app.container import build_container
from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult
from home_optimizer.web.pages import render_dashboard
from home_optimizer.web.schemas import DashboardViewModel, HistoryImportRunResponse

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ClosableGateway(Protocol):
    def close(self) -> None: ...


class HistoryImportRunner(Protocol):
    def import_many(self, request: HistoryImportRequest) -> HistoryImportResult: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...


def _build_history_request(
    settings: AppSettings,
) -> tuple[HistoryImportRequest, int]:
    specs = build_sensor_specs(settings)
    request = HistoryImportRequest.from_settings(settings=settings, specs=specs)
    return request, len(specs)


def create_app(
    settings: AppSettings,
    container_factory: Callable[[AppSettings], WebAppContainer] = build_container,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = container_factory(settings)
        app.state.container = container
        try:
            yield
        finally:
            container.home_assistant.close()

    app = FastAPI(
        title="Home Optimizer API",
        version="0.10",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_container() -> WebAppContainer:
        return app.state.container

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

        request, sensor_count = _build_history_request(settings)
        if sensor_count == 0:
            raise HTTPException(status_code=400, detail="Geen sensoren geconfigureerd voor import.")

        container = get_container()
        result = container.history_import_service.import_many(request)
        total_rows = sum(result.imported_rows.values())

        LOGGER.info(
            "History import completed: %s rows across %s sensors",
            total_rows,
            sensor_count,
        )

        return HistoryImportRunResponse(
            imported_rows=result.imported_rows,
            total_rows=total_rows,
            sensor_count=sensor_count,
        )

    return app
