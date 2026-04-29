from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import identification_response, stored_identified_model_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import (
    IdentificationResponse,
    IdentificationTrainRequest,
    StoredIdentifiedModelResponse,
)

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
DateTimeQuery = Annotated[datetime, Query()]
TrainBody = Annotated[IdentificationTrainRequest, Body()]


def create_identification_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/identification", response_model=IdentificationResponse)
    def get_identification(
        start_time: DateTimeQuery,
        end_time: DateTimeQuery,
        container: ContainerDependency,
        interval_minutes: int = Query(default=15, ge=1),
        train_fraction: float = Query(default=0.8, gt=0.0, lt=1.0),
    ) -> IdentificationResponse:
        try:
            result = container.identification_service.identify(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                train_fraction=train_fraction,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return identification_response(result)

    @router.post("/api/identification/train", response_model=StoredIdentifiedModelResponse)
    def post_identification_train(
        request: TrainBody,
        container: ContainerDependency,
    ) -> StoredIdentifiedModelResponse:
        try:
            model = container.identification_service.identify_and_store(
                start_time=request.start_time,
                end_time=request.end_time,
                interval_minutes=request.interval_minutes,
                train_fraction=request.train_fraction,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return stored_identified_model_response(model)

    return router
