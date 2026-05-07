from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import exp, log, pi, sin

import pytest

from home_optimizer.features.identification import (
    IdentificationDataset,
    IdentificationDatasetRow,
    RoomThermalModelService,
)


def _mass_decay(interval_minutes: int, half_life_hours: float) -> float:
    return exp(-log(2.0) * (interval_minutes / 60.0) / half_life_hours)


def build_synthetic_room_dataset(
    *,
    interval_minutes: int = 15,
    rows: int = 480,
    mass_half_life_hours: float = 8.0,
) -> IdentificationDataset:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    mass_decay = _mass_decay(interval_minutes, mass_half_life_hours)

    intercept = 0.12
    room_coeff = 0.72
    mass_coeff = 0.20
    outdoor_coeff = 0.015
    thermal_coeff = 0.11
    solar_coeff = 0.0012

    room_temperature = 20.0
    mass_temperature = 20.0
    dataset_rows: list[IdentificationDatasetRow] = []

    for index in range(rows):
        timestamp = start + timedelta(minutes=interval_minutes * index)
        hour = timestamp.hour + timestamp.minute / 60.0
        outdoor_temperature = 8.0 + 5.0 * sin(2.0 * pi * index / 96.0)
        solar_gain = max(0.0, 450.0 * sin(pi * (hour - 6.0) / 12.0))
        thermal_output = 0.5 + 2.0 * float(hour < 8.0 or hour >= 19.0)

        dataset_rows.append(
            IdentificationDatasetRow(
                timestamp_utc=timestamp,
                room_temperature_c=room_temperature,
                outdoor_temperature_c=outdoor_temperature,
                thermal_output_estimate_kw=thermal_output,
                solar_gain_proxy_w_m2=solar_gain,
                mode_space=int(thermal_output > 1.0),
                mode_dhw=0,
                mode_off=int(thermal_output <= 1.0),
                is_valid_for_room_identification=True,
            )
        )

        next_room = (
            intercept
            + room_coeff * room_temperature
            + mass_coeff * mass_temperature
            + outdoor_coeff * outdoor_temperature
            + thermal_coeff * thermal_output
            + solar_coeff * solar_gain
        )
        mass_temperature = mass_decay * mass_temperature + (1.0 - mass_decay) * next_room
        room_temperature = next_room

    return IdentificationDataset(
        interval_minutes=interval_minutes,
        start_time_utc=dataset_rows[0].timestamp_utc,
        end_time_utc=dataset_rows[-1].timestamp_utc + timedelta(minutes=interval_minutes),
        rows=dataset_rows,
    )


def test_room_thermal_model_service_fits_and_validates_multi_horizon_rollouts() -> None:
    dataset = build_synthetic_room_dataset()

    result = RoomThermalModelService().fit_and_validate(dataset)

    assert result.model.interval_minutes == 15
    assert result.model.training_sample_count > 100
    assert result.validation.total_valid_rows == 480
    assert result.validation.training_rows > result.validation.holdout_rows
    assert [metric.horizon_minutes for metric in result.validation.metrics] == [60, 360, 1440]

    metrics_by_horizon = {
        metric.horizon_minutes: metric for metric in result.validation.metrics
    }
    assert metrics_by_horizon[60].sample_count > 0
    assert metrics_by_horizon[360].sample_count > 0
    assert metrics_by_horizon[1440].sample_count > 0
    assert metrics_by_horizon[60].mae_c < 0.02
    assert metrics_by_horizon[360].mae_c < 0.05
    assert metrics_by_horizon[1440].mae_c < 0.08


def test_room_thermal_model_service_requires_enough_valid_rows() -> None:
    dataset = build_synthetic_room_dataset(rows=8)

    with pytest.raises(ValueError, match="at least 12 valid room-identification rows"):
        RoomThermalModelService().fit_and_validate(dataset)
