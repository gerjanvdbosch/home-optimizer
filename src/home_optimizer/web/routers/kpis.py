from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.kpi.service import DailyKpiService
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import daily_kpi_response
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.schemas import DailyKpiResponse

ChartDateQuery = Annotated[date, Query(alias="date")]
ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]


def create_kpi_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/kpis", response_model=DailyKpiResponse)
    def get_dashboard_kpis(
        chart_date: ChartDateQuery,
        container: ContainerDependency,
    ) -> DailyKpiResponse:
        return daily_kpi_response(
            DailyKpiService(
                container.time_series_read_repository,
                settings,
            ).get_day_kpis(chart_date)
        )

    return router
