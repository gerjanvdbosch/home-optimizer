from __future__ import annotations

from typing import Annotated
from typing import Literal

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
ControlModeQuery = Annotated[
    Literal["hierarchical_preheat"] | None,
    Query(alias="mpc_control_mode"),
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
        control_mode=plan.control_mode,
        status=plan.status,
        termination_condition=plan.termination_condition,
        feasible=plan.feasible,
        objective_value=plan.objective_value,
        solve_time_seconds=plan.solve_time_seconds,
        heating_explanation=plan.heating_explanation,
        objective_breakdown=MpcObjectiveBreakdownResponse(
            comfort_low=plan.objective_breakdown.comfort_low,
            active_comfort_high_cost=plan.objective_breakdown.active_comfort_high,
            passive_comfort_high_cost=plan.objective_breakdown.passive_comfort_high,
            comfort_high=plan.objective_breakdown.comfort_high,
            comfort_total=plan.objective_breakdown.comfort_total,
            tracking_under_target=plan.objective_breakdown.tracking_under_target,
            tracking_over_target=plan.objective_breakdown.tracking_over_target,
            temperature_tracking=plan.objective_breakdown.temperature_tracking,
            energy_cost=plan.objective_breakdown.energy_cost,
            pv_self_consumption_reward=plan.objective_breakdown.pv_self_consumption_reward,
            captured_pv_kwh=plan.objective_breakdown.captured_pv_kwh,
            preheat_budget_shortfall=plan.objective_breakdown.preheat_budget_shortfall,
            unnecessary_heating=plan.objective_breakdown.unnecessary_heating,
            terminal_cost=plan.objective_breakdown.terminal,
            start_penalty=plan.objective_breakdown.start,
            runtime=plan.objective_breakdown.runtime,
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
                economic_target_c=step.economic_target_c,
                useful_preheat_target_c=step.useful_preheat_target_c,
                preheat_active=step.preheat_active,
                preheat_block_id=step.preheat_block_id,
                preheat_opportunity_score=step.preheat_opportunity_score,
                preheat_budget_share_kwh=step.preheat_budget_share_kwh,
                preheat_charge_kwh=step.preheat_charge_kwh,
                preheat_block_budget_kwh=step.preheat_block_budget_kwh,
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
        mpc_control_mode: ControlModeQuery = None,
    ) -> MpcPlanResponse:
        try:
            plan = container.space_heating_mpc_planning_service.plan(
                start_time_utc=start_time,
                model_id=model_id,
                interval_minutes=interval_minutes,
                horizon_steps=horizon_steps,
                default_effective_heating_kw=default_effective_heating_kw,
                max_solver_seconds=max_solver_seconds,
                control_mode=mpc_control_mode or "hierarchical_preheat",
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _plan_response(plan)

    return router
