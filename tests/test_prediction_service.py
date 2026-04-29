from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.domain import (
    IdentifiedModel,
    NumericPoint,
    NumericSeries,
)
from home_optimizer.features.prediction import BuildingTemperaturePredictionService


class FakePredictionReader:
    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        assert names == ["room_temperature"]
        return [
            NumericSeries(
                name="room_temperature",
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T09:45:00+00:00", value=20.0),
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=20.5),
                ],
            )
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        assert names == ["temperature", "gti_living_room_windows"]
        base_time = datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc)
        timestamps = [
            (base_time + timedelta(minutes=15 * step)).isoformat()
            for step in range(1, 5)
        ]
        return [
            NumericSeries(
                name="temperature",
                unit="degC",
                points=[NumericPoint(timestamp=timestamp, value=10.0) for timestamp in timestamps],
            ),
            NumericSeries(
                name="gti_living_room_windows",
                unit="Wm2",
                points=[NumericPoint(timestamp=timestamp, value=100.0) for timestamp in timestamps],
            ),
        ]


class FakeModelRepository:
    def __init__(self, model: IdentifiedModel) -> None:
        self.model = model

    def latest(self, *, model_kind: str) -> IdentifiedModel | None:
        return self.model if self.model.model_kind == model_kind else None


def test_prediction_service_simulates_multiple_steps() -> None:
    model = IdentifiedModel(
        model_kind="room_temperature",
        model_name="linear_1step_room_temperature",
        trained_at_utc=datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc),
        training_start_time_utc=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        training_end_time_utc=datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        sample_count=100,
        train_sample_count=80,
        test_sample_count=20,
        coefficients={
            "previous_room_temperature": 0.9,
            "outdoor_temperature": 0.02,
            "thermostat_setpoint": 0.03,
            "gti_living_room_windows_adjusted": 0.001,
        },
        intercept=0.5,
        train_rmse=0.05,
        test_rmse=0.1,
        target_name="room_temperature",
    )
    service = BuildingTemperaturePredictionService(
        FakePredictionReader(),
        FakeModelRepository(model),
    )

    thermostat_schedule = NumericSeries(
        name="thermostat_setpoint",
        unit="degC",
        points=[
            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:45:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T11:00:00+00:00", value=21.0),
        ],
    )
    shutter_schedule = NumericSeries(
        name="shutter_living_room",
        unit="percent",
        points=[
            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=50.0),
            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=50.0),
            NumericPoint(timestamp="2026-04-28T10:45:00+00:00", value=50.0),
            NumericPoint(timestamp="2026-04-28T11:00:00+00:00", value=50.0),
        ],
    )

    prediction = service.predict(
        start_time=datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc),
        thermostat_schedule=thermostat_schedule,
        shutter_schedule=shutter_schedule,
    )

    assert prediction.model_name == "linear_1step_room_temperature"
    assert [point.timestamp for point in prediction.room_temperature.points] == [
        "2026-04-28T10:15:00+00:00",
        "2026-04-28T10:30:00+00:00",
        "2026-04-28T10:45:00+00:00",
        "2026-04-28T11:00:00+00:00",
    ]
    assert [point.value for point in prediction.room_temperature.points] == pytest.approx(
        [19.83, 19.227, 18.6843, 18.19587]
    )
