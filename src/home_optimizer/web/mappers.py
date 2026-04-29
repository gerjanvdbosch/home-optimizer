from __future__ import annotations

from home_optimizer.app.history_import_jobs import HistoryImportJob
from home_optimizer.domain import IdentifiedModel, NumericPoint, NumericSeries, TextSeries
from home_optimizer.features.identification.schemas import IdentificationResult
from home_optimizer.features.prediction.schemas import RoomTemperaturePrediction
from home_optimizer.web.schemas import (
    ChartPointResponse,
    ChartSeriesResponse,
    ChartTextPointResponse,
    ChartTextSeriesResponse,
    HistoryImportJobResponse,
    IdentificationResponse,
    IdentificationTrainRequest,
    NumericSeriesRequest,
    PredictionResponse,
    StoredIdentifiedModelResponse,
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


def identification_response(result: IdentificationResult) -> IdentificationResponse:
    return IdentificationResponse(
        model_name=result.model_name,
        interval_minutes=result.interval_minutes,
        sample_count=result.sample_count,
        train_sample_count=result.train_sample_count,
        test_sample_count=result.test_sample_count,
        coefficients=result.coefficients,
        intercept=result.intercept,
        train_rmse=result.train_rmse,
        test_rmse=result.test_rmse,
        test_rmse_recursive=result.test_rmse_recursive,
        target_name=result.target_name,
    )


def stored_identified_model_response(model: IdentifiedModel) -> StoredIdentifiedModelResponse:
    return StoredIdentifiedModelResponse(
        model_name=model.model_name,
        trained_at_utc=model.trained_at_utc,
        training_start_time_utc=model.training_start_time_utc,
        training_end_time_utc=model.training_end_time_utc,
        interval_minutes=model.interval_minutes,
        sample_count=model.sample_count,
        train_sample_count=model.train_sample_count,
        test_sample_count=model.test_sample_count,
        coefficients=model.coefficients,
        intercept=model.intercept,
        train_rmse=model.train_rmse,
        test_rmse=model.test_rmse,
        test_rmse_recursive=model.test_rmse_recursive,
        target_name=model.target_name,
    )


def numeric_series_from_request(series: NumericSeriesRequest) -> NumericSeries:
    return NumericSeries(
        name=series.name,
        unit=series.unit,
        points=[
            NumericPoint(timestamp=point.timestamp, value=point.value)
            for point in series.points
        ],
    )


def prediction_response(result: RoomTemperaturePrediction) -> PredictionResponse:
    return PredictionResponse(
        model_name=result.model_name,
        interval_minutes=result.interval_minutes,
        target_name=result.target_name,
        room_temperature=series_response(result.room_temperature),
    )
