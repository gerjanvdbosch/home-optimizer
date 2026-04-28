from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.domain.charts import ChartPoint, ChartSeries
from home_optimizer.features.system_identification import (
    SystemIdentificationError,
    identify_room_temperature_model,
)


def test_identifies_linear_room_temperature_model() -> None:
    start = datetime(2026, 4, 25, tzinfo=timezone.utc)
    outdoor = _series("outdoor_temperature", [8.0 + i * 0.05 for i in range(50)], start)
    heatpump = _series("hp_electric_power", [0.5 if i % 8 < 4 else 1.8 for i in range(50)], start)
    solar = _series(
        "gti_living_room_windows_adjusted",
        [max(0.0, i - 10) * 3.0 for i in range(50)],
        start,
    )

    room_values = [20.0]
    for i in range(49):
        next_room = (
            0.2
            + 0.92 * room_values[i]
            + 0.06 * outdoor.points[i].value
            + 0.12 * heatpump.points[i].value
            + 0.001 * solar.points[i].value
        )
        room_values.append(next_room)
    room = _series("room_temperature", room_values, start)

    result = identify_room_temperature_model(
        room,
        outdoor,
        heatpump,
        solar,
        sample_interval_minutes=15,
    )

    assert result.target_name == "room_temperature_next"
    assert result.input_names == [
        "room_temperature",
        "outdoor_temperature",
        "hp_electric_power",
        "gti_living_room_windows_adjusted",
    ]
    assert result.metrics.sample_count == 49
    assert result.metrics.rmse == pytest.approx(0.0, abs=1e-10)
    assert result.metrics.r_squared == pytest.approx(1.0)
    assert result.coefficients.intercept == pytest.approx(0.2)
    assert result.coefficients.room_temperature == pytest.approx(0.92)
    assert result.coefficients.outdoor_temperature == pytest.approx(0.06)
    assert result.coefficients.heatpump_power == pytest.approx(0.12)
    assert result.coefficients.solar_gain == pytest.approx(0.001)


def test_rejects_when_too_few_samples_are_aligned() -> None:
    start = datetime(2026, 4, 25, tzinfo=timezone.utc)
    room = _series("room_temperature", [20.0, 20.1, 20.2], start)
    outdoor = _series("outdoor_temperature", [8.0, 8.1, 8.2], start)
    heatpump = _series("hp_electric_power", [1.0, 1.1, 1.2], start)

    with pytest.raises(SystemIdentificationError):
        identify_room_temperature_model(room, outdoor, heatpump)


def _series(name: str, values: list[float], start: datetime) -> ChartSeries:
    return ChartSeries(
        name=name,
        unit=None,
        points=[
            ChartPoint(
                timestamp=(start + timedelta(minutes=15 * i)).isoformat(),
                value=value,
            )
            for i, value in enumerate(values)
        ],
    )
