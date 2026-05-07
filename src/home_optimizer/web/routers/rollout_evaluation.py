from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.identification import (
    IdentificationDatasetService,
    PersistenceTemperaturePredictor,
    RecursiveRolloutEvaluationService,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import temperature_rollout_evaluation_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import TemperatureRolloutEvaluationResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[datetime, Query(alias="start_time")]
EndTimeQuery = Annotated[datetime, Query(alias="end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]
PredictorQuery = Annotated[Literal["persistence"], Query(alias="predictor")]
HorizonHoursQuery = Annotated[list[int], Query(alias="horizon_hours", ge=1)]
DEFAULT_HORIZON_HOURS = [1, 3, 6, 12, 24]


def _dataset_service(
    container: WebAppContainer,
    settings: AppSettings,
) -> IdentificationDatasetService:
    return IdentificationDatasetService(
        container.time_series_read_repository,
        settings,
    )


def _evaluation_service() -> RecursiveRolloutEvaluationService:
    return RecursiveRolloutEvaluationService()


def create_rollout_evaluation_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/rollout-evaluation/room",
        response_model=TemperatureRolloutEvaluationResponse,
    )
    def get_room_rollout_evaluation(
        container: ContainerDependency,
        start_time: StartTimeQuery = datetime(2026, 4, 16, 0, 0, 0, tzinfo=datetime.now().astimezone().tzinfo),
        end_time: EndTimeQuery = datetime(2026, 5, 5, 23, 59, 0, tzinfo=datetime.now().astimezone().tzinfo),
        interval_minutes: IntervalQuery = 15,
        predictor: PredictorQuery = "persistence",
        horizon_hours: HorizonHoursQuery | None = None,
    ) -> TemperatureRolloutEvaluationResponse:
        resolved_horizon_hours = horizon_hours or DEFAULT_HORIZON_HOURS
        dataset = _dataset_service(container, settings).build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        evaluation = _evaluation_service().evaluate_room(
            dataset,
            predictor=PersistenceTemperaturePredictor(field_name="room_temperature_c"),
            horizon_hours=resolved_horizon_hours,
        )
        return temperature_rollout_evaluation_response(
            evaluation,
            predictor_name=predictor,
        )

    @router.get(
        "/api/rollout-evaluation/dhw",
        response_model=TemperatureRolloutEvaluationResponse,
    )
    def get_dhw_rollout_evaluation(
        container: ContainerDependency,
        start_time: StartTimeQuery = datetime(2026, 4, 16, 0, 0, 0, tzinfo=datetime.now().astimezone().tzinfo),
        end_time: EndTimeQuery = datetime(2026, 5, 5, 23, 59, 0, tzinfo=datetime.now().astimezone().tzinfo),
        interval_minutes: IntervalQuery = 15,
        predictor: PredictorQuery = "persistence",
        horizon_hours: HorizonHoursQuery | None = None,
    ) -> TemperatureRolloutEvaluationResponse:
        resolved_horizon_hours = horizon_hours or DEFAULT_HORIZON_HOURS
        dataset = _dataset_service(container, settings).build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        evaluation = _evaluation_service().evaluate_dhw(
            dataset,
            predictor=PersistenceTemperaturePredictor(field_name="dhw_top_temperature_c"),
            horizon_hours=resolved_horizon_hours,
        )
        return temperature_rollout_evaluation_response(
            evaluation,
            predictor_name=predictor,
        )

    return router
