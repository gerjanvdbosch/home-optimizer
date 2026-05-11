from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.dataset import MpcDataset, MpcDatasetService
from home_optimizer.features.modeling import (
    ROOM_ARX_MODEL_KIND,
    ROOM_RC_MODEL_KIND,
    RoomArxConfig,
    RoomRcConfig,
    RoomModelingService,
    StoredModelVersion,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import room_model_catalog_response, train_room_model_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.query_params import FlexibleDatetime
from home_optimizer.web.schemas import RoomModelCatalogResponse, TrainRoomModelResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[FlexibleDatetime, Query(alias="start_time")]
EndTimeQuery = Annotated[FlexibleDatetime, Query(alias="end_time")]
OptionalStartTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="validation_start_time")]
OptionalEndTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="validation_end_time")]
OptionalTestStartTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="test_start_time")]
OptionalTestEndTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="test_end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]
TrainingWindowQuery = Annotated[int | None, Query(alias="training_window_rows", gt=1)]
ValidationWindowQuery = Annotated[int, Query(alias="validation_window_rows", gt=1)]
MinTrainRowsQuery = Annotated[int, Query(alias="min_train_rows", gt=1)]
ActivateQuery = Annotated[bool, Query(alias="activate")]
ModelTypeQuery = Annotated[str, Query(alias="model_type")]


def _slice_dataset(dataset: MpcDataset, start_index: int, end_exclusive: int) -> MpcDataset:
    rows = dataset.rows[start_index:end_exclusive]
    if not rows:
        raise ValueError("dataset split produced an empty slice")
    return MpcDataset(
        interval_minutes=dataset.interval_minutes,
        start_time_utc=rows[0].timestamp_utc,
        end_time_utc=rows[-1].timestamp_utc + timedelta(minutes=dataset.interval_minutes),
        rows=rows,
    )


def _auto_split_dataset(dataset: MpcDataset, config) -> tuple[MpcDataset, MpcDataset, MpcDataset]:
    total_rows = len(dataset.rows)
    preferred_eval_rows = config.validation_window_rows
    minimum_eval_rows = max(12, min(preferred_eval_rows, 24))
    available_holdout_rows = total_rows - config.min_train_rows
    eval_rows = min(preferred_eval_rows, available_holdout_rows // 2)
    if eval_rows < minimum_eval_rows:
        raise ValueError(
            "dataset is too small for automatic train/validation/test split; "
            "use a longer period or provide explicit validation/test dates"
        )

    test_start = total_rows - eval_rows
    validation_start = total_rows - (2 * eval_rows)
    if validation_start < config.min_train_rows:
        raise ValueError(
            "dataset is too small to keep enough training rows after automatic holdout split"
        )

    return (
        _slice_dataset(dataset, 0, validation_start),
        _slice_dataset(dataset, validation_start, test_start),
        _slice_dataset(dataset, test_start, total_rows),
    )


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
        start_time: StartTimeQuery,
        end_time: EndTimeQuery,
        validation_start_time: OptionalStartTimeQuery = None,
        validation_end_time: OptionalEndTimeQuery = None,
        test_start_time: OptionalTestStartTimeQuery = None,
        test_end_time: OptionalTestEndTimeQuery = None,
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
        elif model_type == ROOM_RC_MODEL_KIND:
            rc_kwargs: dict = dict(
                min_train_rows=min_train_rows,
                min_valid_train_rows=min_train_rows,
                training_window_rows=training_window_rows,
                validation_window_rows=validation_window_rows,
            )
            if settings.living_room_glass_area_m2 is not None:
                rc_kwargs["glass_area_m2"] = settings.living_room_glass_area_m2
            config = RoomRcConfig(**rc_kwargs)
        else:
            raise HTTPException(status_code=400, detail=f"unsupported room model type: {model_type}")

        try:
            if (validation_start_time is None) != (validation_end_time is None):
                raise ValueError("validation_start_time and validation_end_time must both be provided")
            if (test_start_time is None) != (test_end_time is None):
                raise ValueError("test_start_time and test_end_time must both be provided")
            if start_time >= end_time:
                raise ValueError("start_time must be before end_time")
            if validation_start_time is not None and validation_start_time >= validation_end_time:
                raise ValueError("validation_start_time must be before validation_end_time")
            if test_start_time is not None and test_start_time >= test_end_time:
                raise ValueError("test_start_time must be before test_end_time")

            dataset = dataset_service.build_dataset(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
            )
            if validation_start_time is not None and validation_end_time is not None:
                train_dataset = dataset
                validation_dataset = dataset_service.build_dataset(
                    start_time=validation_start_time,
                    end_time=validation_end_time,
                    interval_minutes=interval_minutes,
                )
            else:
                train_dataset, validation_dataset, auto_test_dataset = _auto_split_dataset(
                    dataset, config
                )
                test_start_time = auto_test_dataset.start_time_utc
                test_end_time = min(auto_test_dataset.end_time_utc, end_time)

            model = modeling_service.fit_room_model(train_dataset, config=config)
            validation_report = modeling_service.validation_report_for_model(
                validation_dataset,
                model=model,
            )

            test_report = None
            if test_start_time is not None and test_end_time is not None:
                if validation_start_time is None and validation_end_time is None:
                    test_dataset = auto_test_dataset
                else:
                    test_dataset = dataset_service.build_dataset(
                        start_time=test_start_time,
                        end_time=test_end_time,
                        interval_minutes=interval_minutes,
                    )
                test_report = modeling_service.validation_report_for_model(
                    test_dataset,
                    model=model,
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
        return train_room_model_response(
            version,
            validation_report,
            validation_from_utc=(
                validation_dataset.start_time_utc if validation_dataset is not None else None
            ),
            validation_to_utc=(
                validation_dataset.end_time_utc if validation_dataset is not None else None
            ),
            test_report=test_report,
            test_from_utc=test_start_time,
            test_to_utc=test_end_time,
        )

    return router
