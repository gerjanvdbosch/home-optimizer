from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import Response
from starlette.staticfiles import StaticFiles

from home_optimizer.app.container_factories import build_home_assistant_container
from home_optimizer.app.history_import_jobs import HistoryImportJobRunner
from home_optimizer.app.settings import AppSettings
from home_optimizer.web.cache import NO_CACHE_HEADERS
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.routers.dashboard import create_dashboard_router
from home_optimizer.web.routers.history_import import create_history_import_router
from home_optimizer.web.routers.identification import create_identification_router
from home_optimizer.web.routers.prediction import create_prediction_router
from home_optimizer.web.routers.simulation import create_simulation_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE_HEADERS)
        return response


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
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_dashboard_router(settings))
    app.include_router(create_simulation_router(settings))
    app.include_router(create_history_import_router(settings))
    app.include_router(create_identification_router())
    app.include_router(create_prediction_router())

    return app
