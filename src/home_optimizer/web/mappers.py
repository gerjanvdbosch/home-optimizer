from __future__ import annotations

from home_optimizer.app.history_import_jobs import HistoryImportJob
from home_optimizer.domain import (
    BaselineKpiSummary,
    DailyKpis,
    NumericSeries,
    TextSeries,
)
from home_optimizer.features.identification import (
    IdentificationDataset,
    IdentificationDatasetSummary,
)
from home_optimizer.features.modeling import (
    RoomModelValidationReport,
    StoredRoomModelVersion,
)
from home_optimizer.features.simulation import RoomSimulationResult
from home_optimizer.web.schemas import (
    BaselineKpiSummaryResponse,
    ChartPointResponse,
    ChartSeriesResponse,
    ChartTextPointResponse,
    ChartTextSeriesResponse,
    DailyKpiResponse,
    HistoryImportJobResponse,
    HorizonMetricResponse,
    IdentificationDatasetResponse,
    IdentificationDatasetRowResponse,
    IdentificationDatasetSummaryResponse,
    RoomSimulationResponse,
    SegmentValidationResponse,
    TrainRoomModelResponse,
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


def series_response(series: NumericSeries) -> ChartSeriesResponse:
    return ChartSeriesResponse(
        name=series.name,
        unit=series.unit,
        points=[
            ChartPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def text_series_response(series: TextSeries) -> ChartTextSeriesResponse:
    return ChartTextSeriesResponse(
        name=series.name,
        points=[
            ChartTextPointResponse(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def daily_kpi_response(kpis: DailyKpis) -> DailyKpiResponse:
    return DailyKpiResponse(**kpis.model_dump())


def baseline_kpi_summary_response(summary: BaselineKpiSummary) -> BaselineKpiSummaryResponse:
    return BaselineKpiSummaryResponse(**summary.model_dump())


def identification_dataset_summary_response(
    summary: IdentificationDatasetSummary,
) -> IdentificationDatasetSummaryResponse:
    return IdentificationDatasetSummaryResponse(**summary.model_dump())


def identification_dataset_response(
    dataset: IdentificationDataset,
    summary: IdentificationDatasetSummary,
) -> IdentificationDatasetResponse:
    return IdentificationDatasetResponse(
        interval_minutes=dataset.interval_minutes,
        start_time_utc=dataset.start_time_utc,
        end_time_utc=dataset.end_time_utc,
        summary=identification_dataset_summary_response(summary),
        rows=[
            IdentificationDatasetRowResponse(**row.model_dump())
            for row in dataset.rows
        ],
    )


def train_room_model_response(
    version: StoredRoomModelVersion,
    validation_report: RoomModelValidationReport,
) -> TrainRoomModelResponse:
    return TrainRoomModelResponse(
        model_id=version.model_id,
        model_type=version.model_type,
        created_at_utc=version.created_at_utc,
        trained_from_utc=version.model.trained_from_utc,
        trained_to_utc=version.model.trained_to_utc,
        interval_minutes=version.model.interval_minutes,
        sample_count=version.model.sample_count,
        is_active=version.is_active,
        aggregate_metrics=[
            HorizonMetricResponse(**metric.model_dump())
            for metric in validation_report.aggregate_metrics
        ],
        segment_metrics=[
            SegmentValidationResponse(
                segment_name=segment.segment_name,
                description=segment.description,
                metrics=[
                    HorizonMetricResponse(**metric.model_dump())
                    for metric in segment.metrics
                ],
            )
            for segment in validation_report.segment_metrics
        ],
    )


def room_simulation_response(result: RoomSimulationResult) -> RoomSimulationResponse:
    return RoomSimulationResponse(
        model_id=result.model_id,
        anchor_time_utc=result.anchor_time_utc,
        interval_minutes=result.interval_minutes,
        horizon_steps=result.horizon_steps,
        predicted_room_temperature=series_response(result.predicted_room_temperature),
        actual_room_temperature=series_response(result.actual_room_temperature),
        prediction_error_c=series_response(result.prediction_error_c),
        room_target_min_temperature=series_response(result.room_target_min_temperature),
        room_target_max_temperature=series_response(result.room_target_max_temperature),
        outdoor_temperature=series_response(result.outdoor_temperature),
        thermal_output_estimate=series_response(result.thermal_output_estimate),
        solar_irradiance=series_response(result.solar_irradiance),
        solar_gain_proxy=series_response(result.solar_gain_proxy),
        shutter_position=series_response(result.shutter_position),
    )
