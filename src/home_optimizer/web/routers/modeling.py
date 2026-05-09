from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.modeling import (
    ROOM_2R2C_MODEL_KIND,
    ROOM_ARX_MODEL_KIND,
    ROOM_GREYBOX_MODEL_KIND,
    RoomArxConfig,
    RoomGreyBoxConfig,
    Room2R2CConfig,
    RoomModelingService,
    StoredModelVersion,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import room_model_catalog_response, train_room_model_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import RoomModelCatalogResponse, TrainRoomModelResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[datetime, Query(alias="start_time")]
EndTimeQuery = Annotated[datetime, Query(alias="end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]
TrainingWindowQuery = Annotated[int | None, Query(alias="training_window_rows", gt=1)]
ValidationWindowQuery = Annotated[int, Query(alias="validation_window_rows", gt=1)]
MinTrainRowsQuery = Annotated[int, Query(alias="min_train_rows", gt=1)]
ActivateQuery = Annotated[bool, Query(alias="activate")]
ModelTypeQuery = Annotated[str, Query(alias="model_type")]
DEFAULT_TRAIN_START_TIME = datetime(2026, 4, 16, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_TRAIN_END_TIME = datetime(2026, 5, 7, 23, 59, 0, tzinfo=timezone.utc)
DEFAULT_2R2C_TRAINING_WINDOW_ROWS = 30 * 24 * 6


def create_modeling_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/models/room", response_model=RoomModelCatalogResponse)
    def list_room_models(
        container: ContainerDependency,
    ) -> RoomModelCatalogResponse:
        return room_model_catalog_response(
            container.model_version_repository.list_room_model_versions()
        )

    @router.post("/api/train", response_model=TrainRoomModelResponse)
    def train_room_model(
        container: ContainerDependency,
        start_time: StartTimeQuery = DEFAULT_TRAIN_START_TIME,
        end_time: EndTimeQuery = DEFAULT_TRAIN_END_TIME,
        interval_minutes: IntervalQuery = 10,
        training_window_rows: TrainingWindowQuery = None,
        validation_window_rows: ValidationWindowQuery = 144,
        min_train_rows: MinTrainRowsQuery = 96,
        activate: ActivateQuery = False,
        model_type: ModelTypeQuery = ROOM_ARX_MODEL_KIND,
    ) -> TrainRoomModelResponse:
        dataset_service = MpcDatasetService(
            container.dataset_repository,
            settings,
        )
        modeling_service = RoomModelingService()
        if model_type == ROOM_ARX_MODEL_KIND:
            config = RoomArxConfig(
                min_train_rows=min_train_rows,
                training_window_rows=training_window_rows,
                validation_window_rows=validation_window_rows,
            )
        elif model_type == ROOM_2R2C_MODEL_KIND:
            resolved_training_window_rows = (
                training_window_rows
                if training_window_rows is not None
                else DEFAULT_2R2C_TRAINING_WINDOW_ROWS
            )
            config = Room2R2CConfig(
                min_train_rows=min_train_rows,
                training_window_rows=resolved_training_window_rows,
                validation_window_rows=validation_window_rows,
                validation_stride_rows=max(1, (12 * 60) // interval_minutes),
                validation_horizons_steps=[1, 6, 36],
            )
        elif model_type == ROOM_GREYBOX_MODEL_KIND:
            resolved_training_window_rows = (
                training_window_rows
                if training_window_rows is not None
                else DEFAULT_2R2C_TRAINING_WINDOW_ROWS
            )
            config = RoomGreyBoxConfig(
                min_train_rows=min_train_rows,
                training_window_rows=resolved_training_window_rows,
                validation_window_rows=validation_window_rows,
                validation_stride_rows=max(1, (12 * 60) // interval_minutes),
                validation_horizons_steps=[1, 6, 36],
            )
        else:
            raise HTTPException(status_code=400, detail=f"unsupported room model type: {model_type}")

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

        version = StoredModelVersion(
            model_id=f"room-model-{uuid4().hex[:12]}",
            model_type=model_type,
            created_at_utc=datetime.now(timezone.utc),
            is_active=activate,
            model=model,
            validation_report=validation_report,
        )
        container.model_version_repository.save_room_model_version(version)
        return train_room_model_response(version, validation_report)

    return router
