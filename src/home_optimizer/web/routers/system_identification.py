from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.domain.charts import ChartSeries, ChartTextSeries
from home_optimizer.features.system_identification import (
    SystemIdentificationError,
    SystemIdentificationService,
)
from home_optimizer.features.system_identification.schemas import (
    NumericPoint,
    NumericSeries,
    RoomTemperatureModelResult,
    TextPoint,
    TextSeries,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.ports import WebAppContainer

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[datetime, Query(alias="start")]
EndTimeQuery = Annotated[datetime, Query(alias="end")]
SampleIntervalQuery = Annotated[int, Query(alias="sample_interval_minutes", gt=0)]


def create_system_identification_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/system-identification/room-temperature",
        response_model=RoomTemperatureModelResult,
    )
    def identify_room_temperature(
        start_time: StartTimeQuery,
        end_time: EndTimeQuery,
        container: ContainerDependency,
        sample_interval_minutes: SampleIntervalQuery = 15,
    ) -> RoomTemperatureModelResult:
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="end must be after start")

        series = container.dashboard_repository.read_series(
            names=[
                "room_temperature",
                "outdoor_temperature",
                "hp_flow",
                "hp_supply_temperature",
                "hp_return_temperature",
                "defrost_active",
                "booster_heater_active",
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = container.dashboard_repository.read_text_series(
            names=["hp_mode"],
            start_time=start_time,
            end_time=end_time,
        )

        try:
            return SystemIdentificationService(
                sample_interval_minutes=sample_interval_minutes,
            ).identify_room_temperature_model(
                numeric_series=[_numeric_series(item) for item in series],
                text_series=[_text_series(item) for item in text_series],
            )
        except SystemIdentificationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return router


def _numeric_series(series: ChartSeries) -> NumericSeries:
    return NumericSeries(
        name=series.name,
        unit=series.unit,
        points=[
            NumericPoint(
                timestamp=point.timestamp,
                value=point.value,
            )
            for point in series.points
        ],
    )


def _text_series(series: ChartTextSeries) -> TextSeries:
    return TextSeries(
        name=series.name,
        points=[
            TextPoint(
                timestamp=point.timestamp,
                value=point.value,
            )
            for point in series.points
        ],
    )
