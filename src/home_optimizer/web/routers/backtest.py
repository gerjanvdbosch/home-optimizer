from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from home_optimizer.app import AppSettings
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import mpc_backtest_response
from home_optimizer.web.pages import build_dashboard_view_model, render_template
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.query_params import FlexibleDatetime
from home_optimizer.web.schemas import MpcBacktestResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="start_time")]
EndTimeQuery = Annotated[FlexibleDatetime | None, Query(alias="end_time")]
HorizonStepsQuery = Annotated[int, Query(alias="horizon_steps", ge=1, le=288)]
ModelIdQuery = Annotated[str | None, Query(alias="model_id")]
MaxSolverSecondsQuery = Annotated[float | None, Query(alias="max_solver_seconds", gt=0.0)]
ExogenousModeQuery = Annotated[
    Literal["perfect_foresight", "forecast_replay"],
    Query(alias="exogenous_mode"),
]
ControlModeQuery = Annotated[
    Literal["hierarchical_preheat"],
    Query(alias="mpc_control_mode"),
]


def create_backtest_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/backtest", response_class=HTMLResponse)
    def backtest_page() -> HTMLResponse:
        view_model = build_dashboard_view_model(
            settings,
            title="MPC Backtest",
        )
        return HTMLResponse(render_template("backtest.html", view_model))

    @router.get("/api/backtest/space-heating", response_model=MpcBacktestResponse)
    def backtest_space_heating(
        container: ContainerDependency,
        start_time: StartTimeQuery = None,
        end_time: EndTimeQuery = None,
        model_id: ModelIdQuery = None,
        horizon_steps: HorizonStepsQuery = 36,
        max_solver_seconds: MaxSolverSecondsQuery = None,
        exogenous_mode: ExogenousModeQuery = "perfect_foresight",
        mpc_control_mode: ControlModeQuery = "hierarchical_preheat",
    ) -> MpcBacktestResponse:
        if start_time is None:
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            start_time = now - timedelta(days=1)
        if end_time is None:
            end_time = start_time + timedelta(days=1)
        try:
            result = container.space_heating_mpc_backtest_service.run(
                start_time_utc=start_time,
                end_time_utc=end_time,
                model_id=model_id,
                horizon_steps=horizon_steps,
                max_solver_seconds=max_solver_seconds,
                exogenous_mode=exogenous_mode,
                control_mode=mpc_control_mode,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return mpc_backtest_response(result)

    return router
