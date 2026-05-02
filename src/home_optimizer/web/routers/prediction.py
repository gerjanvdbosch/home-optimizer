from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from home_optimizer.domain import ShutterPositionControl
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import (
    mpc_plan_request_from_request,
    mpc_plan_response,
    numeric_series_from_request,
    prediction_comparison_response,
    prediction_response,
    room_temperature_control_inputs_from_request,
)
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import (
    PredictionComparisonResponse,
    MpcPlanRequest,
    MpcPlanResponse,
    PredictionRequest,
    PredictionResponse,
)

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
PredictionBody = Annotated[PredictionRequest, Body()]
MpcPlanBody = Annotated[MpcPlanRequest, Body()]


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
                control_inputs=room_temperature_control_inputs_from_request(
                    request.thermostat_schedule,
                    request.shutter_schedule,
                ),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return prediction_response(result)

    @router.post("/api/prediction/compare", response_model=PredictionComparisonResponse)
    def post_prediction_comparison(
        request: PredictionBody,
        container: ContainerDependency,
    ) -> PredictionComparisonResponse:
        try:
            result = container.prediction_service.predict_vs_actual(
                start_time=request.start_time,
                end_time=request.end_time,
                control_inputs=room_temperature_control_inputs_from_request(
                    request.thermostat_schedule,
                    request.shutter_schedule,
                ),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return prediction_comparison_response(result)

    @router.post("/api/mpc/thermostat-setpoint", response_model=MpcPlanResponse)
    def post_mpc_plan(
        request: MpcPlanBody,
        container: ContainerDependency,
    ) -> MpcPlanResponse:
        try:
            result = container.mpc_planner.propose_plan(
                mpc_plan_request_from_request(request),
                shutter_position=(
                    ShutterPositionControl.from_schedule(numeric_series_from_request(request.shutter_schedule))
                    if request.shutter_schedule is not None
                    else None
                ),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return mpc_plan_response(result)

    return router
