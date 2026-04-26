from __future__ import annotations

from home_optimizer.app.history_import_jobs import HistoryImportJob
from home_optimizer.domain.charts import ChartSeries, ChartTextSeries
from home_optimizer.web.schemas import (
    ChartPointResponse,
    ChartSeriesResponse,
    ChartTextPointResponse,
    ChartTextSeriesResponse,
    HistoryImportJobResponse,
)


def job_response(job: HistoryImportJob) -> HistoryImportJobResponse:
    return HistoryImportJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        imported_rows=job.imported_rows,
        total_rows=job.total_rows,
        sensor_count=job.sensor_count,
        error=job.error,
    )


def series_response(series: ChartSeries) -> ChartSeriesResponse:
    return ChartSeriesResponse(
        name=series.name,
        unit=series.unit,
        points=[
            ChartPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def text_series_response(series: ChartTextSeries) -> ChartTextSeriesResponse:
    return ChartTextSeriesResponse(
        name=series.name,
        points=[
            ChartTextPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )
