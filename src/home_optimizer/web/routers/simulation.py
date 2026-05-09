from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from home_optimizer.app import AppSettings
from home_optimizer.features.simulation import RoomSimulationService
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import room_simulation_response
from home_optimizer.web.pages import build_dashboard_view_model, render_template
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import RoomSimulationResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
AnchorTimeQuery = Annotated[datetime, Query(alias="anchor_time")]
HorizonStepsQuery = Annotated[int, Query(alias="horizon_steps", ge=1, le=288)]
DEFAULT_SIMULATION_ANCHOR_TIME = datetime(2026, 5, 7, 0, 0, 0, tzinfo=timezone.utc)


def create_simulation_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/simulation", response_class=HTMLResponse)
    def simulation_page() -> HTMLResponse:
        view_model = build_dashboard_view_model(
            settings,
            title="Room Simulation",
        )
        return HTMLResponse(render_template("simulation.html", view_model))

    @router.get("/api/simulate/room", response_model=RoomSimulationResponse)
    def simulate_room(
        container: ContainerDependency,
        anchor_time: AnchorTimeQuery = DEFAULT_SIMULATION_ANCHOR_TIME,
        horizon_steps: HorizonStepsQuery = 144,
    ) -> RoomSimulationResponse:
        active_version = container.model_version_repository.get_active_room_model_version()
        if active_version is None:
            raise HTTPException(status_code=404, detail="no active room model found")

        try:
            result = RoomSimulationService(settings).simulate_room(
                samples_reader=container.dataset_repository,
                model_id=active_version.model_id,
                model=active_version.model,
                anchor_time=anchor_time,
                horizon_steps=horizon_steps,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return room_simulation_response(result)

    return router
