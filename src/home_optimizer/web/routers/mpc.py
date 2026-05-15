from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from home_optimizer.app import AppSettings
from home_optimizer.features.mpc import MpcPlan
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.pages import build_dashboard_view_model, render_template
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.query_params import FlexibleDatetime
from home_optimizer.web.schemas import (
    MpcObjectiveBreakdownResponse,
    MpcPlanResponse,
    MpcPlanStepResponse,
    MpcPlanSummaryResponse,
)

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[FlexibleDatetime, Query(alias="start_time")]
HorizonStepsQuery = Annotated[int, Query(alias="horizon_steps", ge=1, le=288)]
IntervalQuery = Annotated[int | None, Query(alias="interval_minutes", ge=1, le=60)]
ModelIdQuery = Annotated[str | None, Query(alias="model_id")]
HeatingKwQuery = Annotated[
    float | None,
    Query(alias="default_effective_heating_kw", ge=0.0),
]
MaxSolverSecondsQuery = Annotated[
    float | None,
    Query(alias="max_solver_seconds", gt=0.0),
]


def _plan_response(plan: MpcPlan) -> MpcPlanResponse:
    summary = MpcPlanSummaryResponse(
        step_count=len(plan.steps),
        start_count=sum(1 for step in plan.steps if step.start),
        stop_count=sum(1 for step in plan.steps if step.stop),
        comfort_violation_count=sum(
            1
            for step in plan.steps
            if step.slack_low_c > 0.0 or step.slack_high_c > 0.0
        ),
        slack_usage_count=sum(
            1
            for step in plan.steps
            if step.slack_low_c > 0.0 or step.slack_high_c > 0.0
        ),
        runtime_steps=sum(1 for step in plan.steps if step.hp_on),
        estimated_energy_cost_eur=sum(step.estimated_energy_cost_eur for step in plan.steps),
    )
    return MpcPlanResponse(
        status=plan.status,
        termination_condition=plan.termination_condition,
        feasible=plan.feasible,
        objective_value=plan.objective_value,
        solve_time_seconds=plan.solve_time_seconds,
        heating_explanation=plan.heating_explanation,
        objective_breakdown=MpcObjectiveBreakdownResponse(
            comfort_low=plan.objective_breakdown.comfort_low,
            comfort_high=plan.objective_breakdown.comfort_high,
            comfort_total=plan.objective_breakdown.comfort_total,
            temperature_tracking=plan.objective_breakdown.temperature_tracking,
            terminal=plan.objective_breakdown.terminal,
            start=plan.objective_breakdown.start,
            runtime=plan.objective_breakdown.runtime,
            energy=plan.objective_breakdown.energy,
            total=plan.objective_breakdown.total,
        ),
        summary=summary,
        steps=[
            MpcPlanStepResponse(
                timestamp_utc=step.timestamp_utc,
                hp_on=step.hp_on,
                start=step.start,
                stop=step.stop,
                predicted_room_temp_c=step.predicted_room_temp_c,
                q_heat_eff_kw=step.q_heat_eff_kw,
                temp_min_c=step.temp_min_c,
                temp_max_c=step.temp_max_c,
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

    @router.get("/mpc", response_class=HTMLResponse)
    def mpc_page() -> HTMLResponse:
        view_model = build_dashboard_view_model(
            settings,
            title="Space-Heating MPC",
        )
        return HTMLResponse(render_template("mpc.html", view_model))

    @router.get("/api/mpc/space-heating/plan", response_model=MpcPlanResponse)
    def plan_space_heating(
        container: ContainerDependency,
        start_time: StartTimeQuery,
        model_id: ModelIdQuery = None,
        horizon_steps: HorizonStepsQuery = 36,
        interval_minutes: IntervalQuery = None,
        default_effective_heating_kw: HeatingKwQuery = None,
        max_solver_seconds: MaxSolverSecondsQuery = None,
    ) -> MpcPlanResponse:
        try:
            plan = container.space_heating_mpc_planning_service.plan(
                start_time_utc=start_time,
                model_id=model_id,
                interval_minutes=interval_minutes,
                horizon_steps=horizon_steps,
                default_effective_heating_kw=default_effective_heating_kw,
                max_solver_seconds=max_solver_seconds,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _plan_response(plan)

    return router
