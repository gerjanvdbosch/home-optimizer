from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.identification.service import IdentificationDatasetService
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import identification_dataset_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import IdentificationDatasetResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[datetime, Query(alias="start_time")]
EndTimeQuery = Annotated[datetime, Query(alias="end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]


def create_identification_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/identification", response_model=IdentificationDatasetResponse)
    def get_identification_dataset(
        container: ContainerDependency,
        start_time: StartTimeQuery = datetime(2026, 2, 8, 0, 0,0, tzinfo=datetime.now().astimezone().tzinfo),
        end_time: EndTimeQuery = datetime(2026, 5, 5, 23, 59, 0, tzinfo=datetime.now().astimezone().tzinfo),
        interval_minutes: IntervalQuery = 15,
    ) -> IdentificationDatasetResponse:
        service = IdentificationDatasetService(
            container.time_series_read_repository,
            settings,
        )
        dataset = service.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        summary = service.summarize_dataset(dataset)
        return identification_dataset_response(dataset, summary)

    return router
