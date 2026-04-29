from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from home_optimizer.app import AppSettings
from home_optimizer.web.pages import build_dashboard_view_model, render_template


def create_simulation_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/simulation", response_class=HTMLResponse)
    def simulation() -> HTMLResponse:
        view_model = build_dashboard_view_model(
            settings,
            title="Home Optimizer Simulatie",
        )
        return HTMLResponse(render_template("simulation.html", view_model))

    return router
