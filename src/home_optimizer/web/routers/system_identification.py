from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.features.system_identification import (
    SystemIdentificationError,
    identify_room_temperature_model,
)
from home_optimizer.features.system_identification.models import ThermalModelIdentificationResult
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
        response_model=ThermalModelIdentificationResult,
    )
    def identify_room_temperature(
        start_time: StartTimeQuery,
        end_time: EndTimeQuery,
        container: ContainerDependency,
        sample_interval_minutes: SampleIntervalQuery = 15,
    ) -> ThermalModelIdentificationResult:
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="end must be after start")

        series = container.dashboard_repository.read_series(
            names=["room_temperature", "outdoor_temperature", "hp_electric_power"],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = {item.name: item for item in series}

        try:
            return identify_room_temperature_model(
                series_by_name["room_temperature"],
                series_by_name["outdoor_temperature"],
                series_by_name["hp_electric_power"],
                sample_interval_minutes=sample_interval_minutes,
            )
        except SystemIdentificationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return router
