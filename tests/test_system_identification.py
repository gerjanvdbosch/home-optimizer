from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.features.system_identification import (
    SystemIdentificationError,
    SystemIdentificationService,
)
from home_optimizer.features.system_identification.schemas import NumericPoint, NumericSeries


def test_identifies_room_temperature_model_with_lags_and_holdout() -> None:
    start = datetime(2026, 4, 25, tzinfo=timezone.utc)
    outdoor = [8.0 + (i % 12) * 0.15 for i in range(90)]
    thermal = [0.4 + (i % 7) * 0.3 + (0.2 if i % 11 == 0 else 0.0) for i in range(90)]
    solar = [max(0.0, 30.0 - abs(i - 45.0)) for i in range(90)]
    room = [20.0, 20.1, 20.2, 20.3]

    for i in range(3, 89):
        heat_loss = room[i] - outdoor[i]
        room.append(
            0.5
            + 0.9 * room[i]
            - 0.02 * heat_loss
            + 0.08 * thermal[i - 1]
            + 0.04 * thermal[i - 2]
            + 0.02 * thermal[i - 4]
            + 0.001 * solar[i - 1]
            + 0.002 * solar[i - 2]
        )

    result = SystemIdentificationService().identify_room_temperature_model(
        numeric_series=[
            _series("room_temperature", room, start, unit="degC"),
            _series("outdoor_temperature", outdoor, start, unit="degC"),
            _series("thermal_output", thermal, start, unit="kW"),
            _series("gti_living_room_windows_adjusted", solar, start, unit="Wm2"),
        ],
    )

    assert result.target_name == "room_temperature_next"
    assert result.input_names == [
        "intercept",
        "room_temperature",
        "room_outdoor_delta",
        "thermal_output_lag_15m",
        "thermal_output_lag_30m",
        "thermal_output_lag_60m",
        "solar_gain_lag_15m",
        "solar_gain_lag_30m",
    ]
    assert result.metrics.train.sample_count == 59
    assert result.metrics.test.sample_count == 26
    assert result.metrics.test.rmse == pytest.approx(0.0, abs=1e-10)
    assert result.metrics.test.r_squared == pytest.approx(1.0)
    assert result.coefficients["room_temperature"] == pytest.approx(0.9)
    assert result.coefficients["room_outdoor_delta"] == pytest.approx(-0.02)
    assert result.coefficients["thermal_output_lag_15m"] == pytest.approx(0.08)
    assert result.coefficients["thermal_output_lag_30m"] == pytest.approx(0.04)
    assert result.coefficients["thermal_output_lag_60m"] == pytest.approx(0.02)
    assert result.coefficients["solar_gain_lag_15m"] == pytest.approx(0.001)
    assert result.coefficients["solar_gain_lag_30m"] == pytest.approx(0.002)
    assert result.actual_series.name == "room_temperature_actual"
    assert result.predicted_series.name == "room_temperature_predicted"
    assert result.residual_series.name == "room_temperature_residual"
    assert len(result.actual_series.points) == 85


def test_room_temperature_model_filters_bad_heatpump_states() -> None:
    start = datetime(2026, 4, 25, tzinfo=timezone.utc)
    values = [20.0 + i * 0.01 for i in range(30)]
    defrost = [0.0 for _ in range(30)]
    defrost[12] = 1.0

    result = SystemIdentificationService().identify_room_temperature_model(
        numeric_series=[
            _series("room_temperature", values, start, unit="degC"),
            _series("outdoor_temperature", [8.0 for _ in range(30)], start, unit="degC"),
            _series("thermal_output", [1.0 for _ in range(30)], start, unit="kW"),
            _series("defrost_active", defrost, start, unit="bool"),
        ],
    )

    assert result.metrics.train.sample_count + result.metrics.test.sample_count == 24


def test_rejects_when_too_few_samples_are_aligned() -> None:
    start = datetime(2026, 4, 25, tzinfo=timezone.utc)

    with pytest.raises(SystemIdentificationError):
        SystemIdentificationService().identify_room_temperature_model(
            numeric_series=[
                _series("room_temperature", [20.0, 20.1, 20.2], start),
                _series("outdoor_temperature", [8.0, 8.1, 8.2], start),
                _series("thermal_output", [1.0, 1.1, 1.2], start),
            ],
        )


def _series(
    name: str,
    values: list[float],
    start: datetime,
    unit: str | None = None,
) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=unit,
        points=[
            NumericPoint(
                timestamp=(start + timedelta(minutes=15 * i)).isoformat(),
                value=value,
            )
            for i, value in enumerate(values)
        ],
    )
