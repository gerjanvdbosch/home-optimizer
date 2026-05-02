from __future__ import annotations

from home_optimizer.app.history_import_jobs import HistoryImportJob
from home_optimizer.domain import (
    IdentifiedModel,
    NumericPoint,
    NumericSeries,
    ShutterPositionControl,
    TextSeries,
    ThermostatSetpointControl,
    normalize_utc_timestamp,
)
from home_optimizer.features.backtesting.metrics import prediction_error_summary
from home_optimizer.features.identification.schemas import IdentificationResult
from home_optimizer.features.mpc.schemas import (
    ThermostatSetpointCandidateEvaluation,
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointMpcPlanRequest,
)
from home_optimizer.features.prediction.schemas import (
    RoomTemperatureControlInputs,
    RoomTemperaturePrediction,
    RoomTemperaturePredictionComparison,
)
from home_optimizer.web.schemas import (
    ChartPointResponse,
    ChartSeriesResponse,
    ChartTextPointResponse,
    ChartTextSeriesResponse,
    HistoryImportJobResponse,
    IdentificationResponse,
    NumericSeriesRequest,
    ModelTrainingRunResponse,
    MpcCandidateResponse,
    MpcPlanRequest,
    MpcPlanResponse,
    PredictionComparisonResponse,
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


def model_training_run_response(models: list[IdentifiedModel]) -> ModelTrainingRunResponse:
    return ModelTrainingRunResponse(
        models=[stored_identified_model_response(model) for model in models]
    )


def numeric_series_from_request(series: NumericSeriesRequest) -> NumericSeries:
    return NumericSeries(
        name=series.name,
        unit=series.unit,
        points=[
            NumericPoint(timestamp=normalize_utc_timestamp(point.timestamp), value=point.value)
            for point in series.points
        ],
    )


def room_temperature_control_inputs_from_request(
    thermostat_schedule: NumericSeriesRequest,
    shutter_schedule: NumericSeriesRequest | None = None,
) -> RoomTemperatureControlInputs:
    return RoomTemperatureControlInputs(
        thermostat_setpoint=ThermostatSetpointControl.from_schedule(
            numeric_series_from_request(thermostat_schedule)
        ),
        shutter_position=(
            ShutterPositionControl.from_schedule(numeric_series_from_request(shutter_schedule))
            if shutter_schedule is not None
            else None
        ),
    )


def mpc_plan_request_from_request(request: MpcPlanRequest) -> ThermostatSetpointMpcPlanRequest:
    return ThermostatSetpointMpcPlanRequest(
        start_time=request.start_time,
        end_time=request.end_time,
        interval_minutes=request.interval_minutes,
        allowed_setpoints=request.allowed_setpoints,
        switch_times=request.switch_times,
        comfort_min_temperature=request.comfort_min_temperature,
        comfort_max_temperature=request.comfort_max_temperature,
        setpoint_change_penalty=request.setpoint_change_penalty,
    )


def prediction_response(result: RoomTemperaturePrediction) -> PredictionResponse:
    return PredictionResponse(
        model_name=result.model_name,
        interval_minutes=result.interval_minutes,
        target_name=result.target_name,
        room_temperature=series_response(result.room_temperature),
    )


def prediction_comparison_response(
    result: RoomTemperaturePredictionComparison,
) -> PredictionComparisonResponse:
    overlap_count, rmse, bias, max_absolute_error = prediction_error_summary(
        predicted=result.predicted_room_temperature,
        actual=result.actual_room_temperature,
    )
    return PredictionComparisonResponse(
        model_name=result.model_name,
        interval_minutes=result.interval_minutes,
        target_name=result.target_name,
        predicted_room_temperature=series_response(result.predicted_room_temperature),
        actual_room_temperature=series_response(result.actual_room_temperature),
        overlap_count=overlap_count,
        rmse=rmse,
        bias=bias,
        max_absolute_error=max_absolute_error,
    )


def mpc_candidate_response(
    result: ThermostatSetpointCandidateEvaluation,
) -> MpcCandidateResponse:
    return MpcCandidateResponse(
        candidate_name=result.candidate_name,
        thermostat_setpoint_schedule=series_response(result.thermostat_setpoint_schedule),
        predicted_room_temperature=series_response(result.predicted_room_temperature),
        total_cost=result.total_cost,
        comfort_violation_cost=result.comfort_violation_cost,
        setpoint_change_cost=result.setpoint_change_cost,
        minimum_predicted_temperature=result.minimum_predicted_temperature,
        maximum_predicted_temperature=result.maximum_predicted_temperature,
    )


def mpc_plan_response(result: ThermostatSetpointMpcEvaluationResult) -> MpcPlanResponse:
    candidates = [mpc_candidate_response(item) for item in result.candidate_results]
    best_candidate = next(
        candidate for candidate in candidates if candidate.candidate_name == result.best_candidate.candidate_name
    )
    return MpcPlanResponse(
        model_name=result.model_name,
        interval_minutes=result.interval_minutes,
        candidate_results=candidates,
        best_candidate=best_candidate,
    )
