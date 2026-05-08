from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.modeling import (
    RoomModelConfig,
    RoomModelingService,
    StoredRoomModelVersion,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import train_room_model_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import TrainRoomModelResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[datetime, Query(alias="start_time")]
EndTimeQuery = Annotated[datetime, Query(alias="end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]
TrainingWindowQuery = Annotated[int | None, Query(alias="training_window_rows", gt=1)]
ValidationWindowQuery = Annotated[int, Query(alias="validation_window_rows", gt=1)]
MinTrainRowsQuery = Annotated[int, Query(alias="min_train_rows", gt=1)]
ActivateQuery = Annotated[bool, Query(alias="activate")]
DEFAULT_TRAIN_START_TIME = datetime(2026, 4, 16, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_TRAIN_END_TIME = datetime(2026, 5, 7, 23, 59, 0, tzinfo=timezone.utc)


def create_train_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/train", response_model=TrainRoomModelResponse)
    def train_room_model(
        container: ContainerDependency,
        start_time: StartTimeQuery = DEFAULT_TRAIN_START_TIME,
        end_time: EndTimeQuery = DEFAULT_TRAIN_END_TIME,
        interval_minutes: IntervalQuery = 10,
        training_window_rows: TrainingWindowQuery = None,
        validation_window_rows: ValidationWindowQuery = 144,
        min_train_rows: MinTrainRowsQuery = 96,
        activate: ActivateQuery = False,
    ) -> TrainRoomModelResponse:
        dataset_service = MpcDatasetService(
            container.time_series_read_repository,
            settings,
        )
        modeling_service = RoomModelingService()
        config = RoomModelConfig(
            min_train_rows=min_train_rows,
            training_window_rows=training_window_rows,
            validation_window_rows=validation_window_rows,
        )

        try:
            dataset = dataset_service.build_dataset(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
            )
            model = modeling_service.fit_room_model(dataset, config=config)
            validation_report = modeling_service.rolling_validate_room_model(
                dataset,
                config=config,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        version = StoredRoomModelVersion(
            model_id=f"room-model-{uuid4().hex[:12]}",
            created_at_utc=datetime.now(timezone.utc),
            is_active=activate,
            model=model,
            validation_report=validation_report,
        )
        container.model_version_repository.save_room_model_version(version)
        return train_room_model_response(version, validation_report)

    return router
