from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling import RoomArxConfig, RoomModelingService


def build_synthetic_room_dataset(row_count: int = 80) -> MpcDataset:
    start_time = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[MpcDatasetRow] = []
    room_temperature = 20.0

    for index in range(row_count):
        outdoor_temperature = 10.0 + ((index % 12) * 0.1)
        thermal_output = 0.0 if index % 8 < 4 else 2.0
        solar_gain = 0.0 if index % 24 < 10 else 80.0
        occupied_flag = 1 if 7 <= (index % 24) <= 22 else 0
        solar_irradiance = 0.0 if index % 24 < 10 else 250.0
        shutter_position = 100.0 if index % 16 < 8 else 0.0

        rows.append(
            MpcDatasetRow(
                timestamp_utc=start_time + timedelta(minutes=10 * index),
                room_temperature_c=room_temperature,
                outdoor_temperature_c=outdoor_temperature,
                thermal_output_estimate_kw=thermal_output,
                solar_gain_proxy_w_m2=solar_gain,
                solar_irradiance_w_m2=solar_irradiance,
                shutter_position_pct=shutter_position,
                occupied_flag=occupied_flag,
            )
        )
        room_temperature = (
            3.0
            + (0.82 * room_temperature)
            + (0.08 * outdoor_temperature)
            + (0.35 * thermal_output)
            + (0.002 * solar_gain)
            + (0.05 * occupied_flag)
        )

    return MpcDataset(
        interval_minutes=10,
        start_time_utc=rows[0].timestamp_utc,
        end_time_utc=rows[-1].timestamp_utc + timedelta(minutes=10),
        rows=rows,
    )


def test_room_modeling_service_fits_linear_room_model_on_dataset() -> None:
    dataset = build_synthetic_room_dataset()
    service = RoomModelingService()
    config = RoomArxConfig(
        room_temperature_lags=[0],
        outdoor_temperature_lags=[0],
        thermal_output_lags=[0],
        solar_gain_lags=[0],
        shutter_position_lags=[0],
        solar_shutter_interaction_lags=[0],
        occupied_flag_lags=[0],
        ridge_alpha=0.0,
        min_train_rows=20,
        validation_window_rows=10,
        validation_horizons_steps=[1, 3],
    )

    model = service.fit_room_model(dataset, config=config)
    prediction = service.predict_next_room_temperature(model, dataset.rows, source_index=30)

    assert model.sample_count > 50
    assert model.feature_names == [
        "room_temperature_lag_0",
        "outdoor_temperature_lag_0",
        "thermal_output_lag_0",
        "solar_gain_lag_0",
        "shutter_position_lag_0",
        "solar_shutter_interaction_lag_0",
        "occupied_flag_lag_0",
    ]
    assert prediction is not None
    assert prediction == pytest.approx(dataset.rows[31].room_temperature_c, abs=1e-8)


def test_room_modeling_service_runs_rolling_recursive_validation() -> None:
    dataset = build_synthetic_room_dataset()
    service = RoomModelingService()
    config = RoomArxConfig(
        room_temperature_lags=[0],
        outdoor_temperature_lags=[0],
        thermal_output_lags=[0],
        solar_gain_lags=[0],
        shutter_position_lags=[0],
        solar_shutter_interaction_lags=[0],
        occupied_flag_lags=[0],
        ridge_alpha=0.0,
        min_train_rows=24,
        validation_window_rows=12,
        validation_horizons_steps=[1, 3, 6],
    )

    report = service.rolling_validate_room_model(dataset, config=config)

    assert len(report.folds) >= 2
    assert [metric.horizon_steps for metric in report.aggregate_metrics] == [1, 3, 6]
    assert report.aggregate_metrics[0].mae_c is not None
    assert report.aggregate_metrics[0].mae_c < 1e-6
    assert report.aggregate_metrics[1].mae_c is not None
    assert report.aggregate_metrics[1].mae_c < 1e-5
    assert [segment.segment_name for segment in report.segment_metrics] == [
        "sunny",
        "heating_active",
        "shutters_open",
        "shutters_closed",
        "sunny_midday",
    ]


def test_room_modeling_service_improves_24h_sample_count_and_segment_bias_metrics() -> None:
    dataset = build_synthetic_room_dataset(row_count=420)
    service = RoomModelingService()
    config = RoomArxConfig(
        room_temperature_lags=[0],
        outdoor_temperature_lags=[0],
        thermal_output_lags=[0],
        solar_gain_lags=[0],
        shutter_position_lags=[0],
        solar_shutter_interaction_lags=[0],
        occupied_flag_lags=[0],
        ridge_alpha=0.0,
        min_train_rows=60,
        validation_window_rows=144,
        validation_stride_rows=6,
        validation_horizons_steps=[144],
    )

    report = service.rolling_validate_room_model(dataset, config=config)

    assert report.aggregate_metrics[0].horizon_minutes == 1440
    assert report.aggregate_metrics[0].sample_count > 20
    sunny_midday = next(
        segment for segment in report.segment_metrics if segment.segment_name == "sunny_midday"
    )
    assert sunny_midday.metrics[0].sample_count > 0
    assert sunny_midday.metrics[0].bias_c is not None
