from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import (
    numeric_series_from_request,
    prediction_response,
)
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import PredictionRequest, PredictionResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
PredictionBody = Annotated[PredictionRequest, Body()]


def create_prediction_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/prediction", response_model=PredictionResponse)
    def post_prediction(
        request: PredictionBody,
        container: ContainerDependency,
    ) -> PredictionResponse:
        try:
            result = container.prediction_service.predict(
                start_time=request.start_time,
                end_time=request.end_time,
                thermostat_schedule=numeric_series_from_request(request.thermostat_schedule),
                shutter_schedule=(
                    numeric_series_from_request(request.shutter_schedule)
                    if request.shutter_schedule is not None
                    else None
                ),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return prediction_response(result)

    return router
