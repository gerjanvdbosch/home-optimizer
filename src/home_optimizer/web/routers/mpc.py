from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.mpc import MpcPlan
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.query_params import FlexibleDatetime
from home_optimizer.web.schemas import MpcPlanResponse, MpcPlanStepResponse

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[FlexibleDatetime, Query(alias="start_time")]
HorizonStepsQuery = Annotated[int, Query(alias="horizon_steps", ge=1, le=288)]
IntervalQuery = Annotated[int | None, Query(alias="interval_minutes", ge=1, le=60)]
HeatingKwQuery = Annotated[
    float | None,
    Query(alias="default_effective_heating_kw", ge=0.0),
]
MaxSolverSecondsQuery = Annotated[
    float | None,
    Query(alias="max_solver_seconds", gt=0.0),
]


def _plan_response(plan: MpcPlan) -> MpcPlanResponse:
    return MpcPlanResponse(
        status=plan.status,
        termination_condition=plan.termination_condition,
        feasible=plan.feasible,
        objective_value=plan.objective_value,
        solve_time_seconds=plan.solve_time_seconds,
        steps=[
            MpcPlanStepResponse(
                timestamp_utc=step.timestamp_utc,
                hp_on=step.hp_on,
                start=step.start,
                stop=step.stop,
                predicted_room_temp_c=step.predicted_room_temp_c,
                slack_low_c=step.slack_low_c,
                slack_high_c=step.slack_high_c,
                effective_heating_kw=step.effective_heating_kw,
                price_eur_kwh=step.price_eur_kwh,
                estimated_energy_cost_eur=step.estimated_energy_cost_eur,
            )
            for step in plan.steps
        ],
    )


def create_mpc_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/mpc/space-heating/plan", response_model=MpcPlanResponse)
    def plan_space_heating(
        container: ContainerDependency,
        start_time: StartTimeQuery,
        horizon_steps: HorizonStepsQuery = 36,
        interval_minutes: IntervalQuery = None,
        default_effective_heating_kw: HeatingKwQuery = None,
        max_solver_seconds: MaxSolverSecondsQuery = None,
    ) -> MpcPlanResponse:
        try:
            plan = container.space_heating_mpc_planning_service.plan(
                start_time_utc=start_time,
                interval_minutes=interval_minutes,
                horizon_steps=horizon_steps,
                default_effective_heating_kw=default_effective_heating_kw,
                max_solver_seconds=max_solver_seconds,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _plan_response(plan)

    return router
