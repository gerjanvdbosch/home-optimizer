from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from home_optimizer.app import AppSettings
from home_optimizer.features.identification.service import (
    DailyKpiService,
    IdentificationDatasetService,
)
from home_optimizer.web.dependencies import get_container
from home_optimizer.web.mappers import (
    baseline_kpi_summary_response,
    daily_kpi_response,
    identification_dataset_response,
)
from home_optimizer.web.ports import WebAppContainer
from home_optimizer.web.query_params import FlexibleDatetime, FlexibleEndDatetime
from home_optimizer.web.schemas import (
    BaselineKpiSummaryResponse,
    DailyKpiResponse,
    IdentificationDatasetResponse,
)

ContainerDependency = Annotated[WebAppContainer, Depends(get_container)]
StartTimeQuery = Annotated[FlexibleDatetime, Query(alias="start_time")]
EndTimeQuery = Annotated[FlexibleEndDatetime, Query(alias="end_time")]
IntervalQuery = Annotated[int, Query(alias="interval_minutes", ge=1, le=60)]
ChartDateQuery = Annotated[date, Query(alias="date")]
SummaryStartDateQuery = Annotated[date, Query(alias="start_date")]
SummaryEndDateQuery = Annotated[date, Query(alias="end_date")]


def create_identification_router(settings: AppSettings) -> APIRouter:
    router = APIRouter()

    @router.get("/api/identification", response_model=IdentificationDatasetResponse)
    def get_identification_dataset(
        container: ContainerDependency,
        start_time: StartTimeQuery,
        end_time: EndTimeQuery,
        interval_minutes: IntervalQuery = 15,
    ) -> IdentificationDatasetResponse:
        service = IdentificationDatasetService(container.dataset_repository, settings)
        dataset = service.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        summary = service.summarize_dataset(dataset)
        return identification_dataset_response(dataset, summary)

    @router.get("/api/kpis", response_model=DailyKpiResponse)
    def get_dashboard_kpis(
        chart_date: ChartDateQuery,
        container: ContainerDependency,
    ) -> DailyKpiResponse:
        return daily_kpi_response(
            DailyKpiService(container.time_series_read_repository, settings).get_day_kpis(
                chart_date
            )
        )

    @router.get("/api/kpi-summary", response_model=BaselineKpiSummaryResponse)
    def get_baseline_kpi_summary(
        container: ContainerDependency,
        start_date: SummaryStartDateQuery | None = None,
        end_date: SummaryEndDateQuery | None = None,
    ) -> BaselineKpiSummaryResponse:
        resolved_end_date = end_date or date.today()
        resolved_start_date = start_date or (resolved_end_date - timedelta(days=89))
        return baseline_kpi_summary_response(
            DailyKpiService(container.time_series_read_repository, settings).get_baseline_summary(
                resolved_start_date, resolved_end_date
            )
        )

    return router
