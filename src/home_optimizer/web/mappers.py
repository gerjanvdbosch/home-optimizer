from __future__ import annotations

from datetime import datetime

from home_optimizer.app.history_import_jobs import HistoryImportJob
from home_optimizer.domain import (
    BaselineKpiSummary,
    DailyKpis,
    NumericSeries,
    TextSeries,
)
from home_optimizer.features.backtest.models import MpcBacktestResult, MpcBacktestSummary
from home_optimizer.features.identification import (
    IdentificationDataset,
    IdentificationDatasetSummary,
)
from home_optimizer.features.modeling import (
    RoomModelValidationReport,
    RoomSimulationResult,
    StoredModelVersion,
    StoredModelVersionSummary,
)
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
    MpcBacktestDeltaResponse,
    MpcBacktestResponse,
    MpcBacktestStepResponse,
    MpcBacktestSummaryResponse,
    MpcObjectiveBreakdownResponse,
    RoomSimulationResponse,
    RoomModelCatalogResponse,
    RoomModelVersionDetailResponse,
    RoomModelVersionSummaryResponse,
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
    version: StoredModelVersion,
    validation_report: RoomModelValidationReport,
    *,
    validation_from_utc: datetime | None = None,
    validation_to_utc: datetime | None = None,
    test_report: RoomModelValidationReport | None = None,
    test_from_utc: datetime | None = None,
    test_to_utc: datetime | None = None,
) -> TrainRoomModelResponse:
    training_metadata = getattr(version.model, "training_metadata", {}) or {}
    return TrainRoomModelResponse(
        model_id=version.model_id,
        model_type=version.model_type,
        created_at_utc=version.created_at_utc,
        trained_from_utc=version.model.trained_from_utc,
        trained_to_utc=version.model.trained_to_utc,
        validation_from_utc=validation_from_utc,
        validation_to_utc=validation_to_utc,
        test_from_utc=test_from_utc,
        test_to_utc=test_to_utc,
        interval_minutes=version.model.interval_minutes,
        sample_count=version.model.sample_count,
        is_active=version.is_active,
        fit_quality=training_metadata.get("fit_quality"),
        fit_quality_reasons=list(training_metadata.get("fit_quality_reasons", [])),
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
        test_aggregate_metrics=[
            HorizonMetricResponse(**metric.model_dump())
            for metric in (test_report.aggregate_metrics if test_report is not None else [])
        ],
        test_segment_metrics=[
            SegmentValidationResponse(
                segment_name=segment.segment_name,
                description=segment.description,
                metrics=[
                    HorizonMetricResponse(**metric.model_dump())
                    for metric in segment.metrics
                ],
            )
            for segment in (test_report.segment_metrics if test_report is not None else [])
        ],
    )


def room_model_version_summary_response(
    summary: StoredModelVersionSummary,
) -> RoomModelVersionSummaryResponse:
    return RoomModelVersionSummaryResponse(**summary.model_dump())


def room_model_catalog_response(
    summaries: list[StoredModelVersionSummary],
) -> RoomModelCatalogResponse:
    return RoomModelCatalogResponse(
        models=[
            room_model_version_summary_response(summary)
            for summary in summaries
        ]
    )


def room_model_version_detail_response(
    version: StoredModelVersion,
) -> RoomModelVersionDetailResponse:
    training_metadata = getattr(version.model, "training_metadata", {}) or {}
    validation_report = version.validation_report
    return RoomModelVersionDetailResponse(
        model_id=version.model_id,
        model_type=version.model_type,
        created_at_utc=version.created_at_utc,
        trained_from_utc=version.model.trained_from_utc,
        trained_to_utc=version.model.trained_to_utc,
        interval_minutes=version.model.interval_minutes,
        sample_count=version.model.sample_count,
        is_active=version.is_active,
        fit_quality=training_metadata.get("fit_quality"),
        fit_quality_reasons=list(training_metadata.get("fit_quality_reasons", [])),
        aggregate_metrics=[
            HorizonMetricResponse(**metric.model_dump())
            for metric in (validation_report.aggregate_metrics if validation_report is not None else [])
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
            for segment in (validation_report.segment_metrics if validation_report is not None else [])
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


def mpc_backtest_summary_response(summary: MpcBacktestSummary) -> MpcBacktestSummaryResponse:
    return MpcBacktestSummaryResponse(**summary.model_dump())


def mpc_backtest_response(result: MpcBacktestResult) -> MpcBacktestResponse:
    mpc_summary = mpc_backtest_summary_response(result.mpc_summary)
    historical_summary = mpc_backtest_summary_response(result.historical_summary)
    executed_breakdown = MpcObjectiveBreakdownResponse(
        comfort_low=result.mpc_objective_breakdown.comfort_low,
        comfort_high=result.mpc_objective_breakdown.comfort_high,
        comfort_total=result.mpc_objective_breakdown.comfort_total,
        tracking_under_target=result.mpc_objective_breakdown.tracking_under_target,
        tracking_over_target=result.mpc_objective_breakdown.tracking_over_target,
        temperature_tracking=result.mpc_objective_breakdown.temperature_tracking,
        energy_cost=result.mpc_objective_breakdown.energy_cost,
        pv_self_consumption_reward=(
            result.mpc_objective_breakdown.pv_self_consumption_reward
        ),
        unnecessary_heating=result.mpc_objective_breakdown.unnecessary_heating,
        terminal_cost=result.mpc_objective_breakdown.terminal,
        start_penalty=result.mpc_objective_breakdown.start,
        runtime=result.mpc_objective_breakdown.runtime,
        total=result.mpc_objective_breakdown.total,
    )
    solver_breakdown = MpcObjectiveBreakdownResponse(
        comfort_low=result.solver_objective_breakdown.comfort_low,
        comfort_high=result.solver_objective_breakdown.comfort_high,
        comfort_total=result.solver_objective_breakdown.comfort_total,
        tracking_under_target=result.solver_objective_breakdown.tracking_under_target,
        tracking_over_target=result.solver_objective_breakdown.tracking_over_target,
        temperature_tracking=result.solver_objective_breakdown.temperature_tracking,
        energy_cost=result.solver_objective_breakdown.energy_cost,
        pv_self_consumption_reward=(
            result.solver_objective_breakdown.pv_self_consumption_reward
        ),
        unnecessary_heating=result.solver_objective_breakdown.unnecessary_heating,
        terminal_cost=result.solver_objective_breakdown.terminal,
        start_penalty=result.solver_objective_breakdown.start,
        runtime=result.solver_objective_breakdown.runtime,
        total=result.solver_objective_breakdown.total,
    )
    return MpcBacktestResponse(
        model_id=result.model_id,
        model_type=result.model_type,
        start_time_utc=result.start_time_utc,
        end_time_utc=result.end_time_utc,
        interval_minutes=result.interval_minutes,
        horizon_steps=result.horizon_steps,
        step_count=len(result.step_results),
        mpc_objective_breakdown=executed_breakdown,
        solver_objective_breakdown=solver_breakdown,
        mpc_summary=mpc_summary,
        historical_summary=historical_summary,
        delta=MpcBacktestDeltaResponse(
            comfort_violation_minutes=(
                result.mpc_summary.comfort_violation_minutes
                - result.historical_summary.comfort_violation_minutes
            ),
            degree_minutes_below_comfort=(
                result.mpc_summary.degree_minutes_below_comfort
                - result.historical_summary.degree_minutes_below_comfort
            ),
            degree_minutes_above_comfort=(
                result.mpc_summary.degree_minutes_above_comfort
                - result.historical_summary.degree_minutes_above_comfort
            ),
            starts_per_day=(
                result.mpc_summary.starts_per_day
                - result.historical_summary.starts_per_day
            ),
            runtime_minutes=(
                result.mpc_summary.runtime_minutes
                - result.historical_summary.runtime_minutes
            ),
            estimated_energy_cost_eur=(
                result.mpc_summary.estimated_energy_cost_eur
                - result.historical_summary.estimated_energy_cost_eur
            ),
            average_solver_runtime_seconds=result.mpc_summary.average_solver_runtime_seconds,
            infeasible_count=result.mpc_summary.infeasible_count,
            slack_usage_count=result.mpc_summary.slack_usage_count,
        ),
        total_solver_runtime_seconds=result.total_solver_runtime_seconds,
        steps=[
            MpcBacktestStepResponse(**step.model_dump())
            for step in result.step_results
        ],
    )
