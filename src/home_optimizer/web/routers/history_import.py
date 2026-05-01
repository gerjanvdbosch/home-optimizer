from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from home_optimizer.app.history_import_jobs import HistoryImportJobRunner
from home_optimizer.app.history_import_requests import build_history_import_request
from home_optimizer.app.settings import AppSettings
from home_optimizer.web.dependencies import get_container, get_history_import_jobs
from home_optimizer.web.mappers import job_response
from home_optimizer.web.schemas import (
    HistoryImportJobResponse,
    HistoryImportRunResponse,
    WeatherImportResponse,
)

LOGGER = logging.getLogger(__name__)
HistoryImportJobsDependency = Annotated[HistoryImportJobRunner, Depends(get_history_import_jobs)]
ContainerDependency = Annotated[object, Depends(get_container)]


def create_history_import_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.post("/api/history-import", response_model=HistoryImportRunResponse)
    def run_history_import(
        history_import_jobs: HistoryImportJobsDependency,
    ) -> HistoryImportRunResponse:
        request = build_history_import_request(settings)
        if len(request.specs) == 0:
            raise HTTPException(status_code=400, detail="Geen sensoren geconfigureerd voor import.")

        job = history_import_jobs.start()
        LOGGER.info("History import job started: %s", job.job_id)

        return HistoryImportRunResponse(
            job_id=job.job_id,
            status=job.status,
            sensor_count=len(request.specs),
        )

    @router.get("/api/history-import/jobs/{job_id}", response_model=HistoryImportJobResponse)
    def get_history_import_job(
        job_id: str,
        history_import_jobs: HistoryImportJobsDependency,
    ) -> HistoryImportJobResponse:
        job = history_import_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Import job niet gevonden.")

        return job_response(job)

    @router.post("/api/weather-import", response_model=WeatherImportResponse)
    def run_weather_import(container: ContainerDependency) -> WeatherImportResponse:
        imported_rows = container.weather_import_service.import_weather_data()
        return WeatherImportResponse(imported_rows=imported_rows)

    return router
