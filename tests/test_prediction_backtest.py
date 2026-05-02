from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from home_optimizer.domain import IdentifiedModel, NumericPoint, NumericSeries
from home_optimizer.features.backtesting import RoomTemperatureBacktestingService
from home_optimizer.features.prediction import RoomTemperaturePredictionService


class FakeBacktestReader:
    def __init__(self) -> None:
        self._series_by_name = self._build_series()

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [self._series_by_name[name] for name in names if name in self._series_by_name]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [self._series_by_name[name] for name in names if name in self._series_by_name]

    @staticmethod
    def _build_series() -> dict[str, NumericSeries]:
        base_time = datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc)
        schedule_points = [
            NumericPoint(
                timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                value=20.0,
            )
            for step in range(0, 96)
        ]
        room_points = [
            NumericPoint(
                timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                value=20.0,
            )
            for step in range(-2, 96)
        ]
        forecast_points = [
            NumericPoint(
                timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                value=10.0,
            )
            for step in range(1, 97)
        ]
        solar_points = [
            NumericPoint(
                timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                value=0.0,
            )
            for step in range(1, 97)
        ]

        return {
            "thermostat_setpoint": NumericSeries(
                name="thermostat_setpoint",
                unit="degC",
                points=schedule_points,
            ),
            "shutter_living_room": NumericSeries(
                name="shutter_living_room",
                unit="percent",
                points=schedule_points,
            ),
            "room_temperature": NumericSeries(
                name="room_temperature",
                unit="degC",
                points=room_points,
            ),
            "temperature": NumericSeries(
                name="temperature",
                unit="degC",
                points=forecast_points,
            ),
            "gti_living_room_windows": NumericSeries(
                name="gti_living_room_windows",
                unit="Wm2",
                points=solar_points,
            ),
        }


class FakeModelRepository:
    def __init__(self, model: IdentifiedModel) -> None:
        self.model = model

    def latest(self, *, model_kind: str) -> IdentifiedModel | None:
        return self.model if self.model.model_kind == model_kind else None


def test_prediction_service_backtests_day_range() -> None:
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
            "previous_room_temperature": 1.0,
            "outdoor_temperature": 0.0,
            "previous_thermostat_setpoint": 0.0,
            "gti_living_room_windows_adjusted": 0.0,
        },
        intercept=0.0,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )
    service = RoomTemperaturePredictionService(
        FakeBacktestReader(),
        FakeModelRepository(model),
    )

    backtesting_service = RoomTemperatureBacktestingService(
        FakeBacktestReader(),
        FakeModelRepository(model),
        service,
    )

    result = backtesting_service.backtest_by_day(
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 28),
        horizon_hours=24,
        timezone_info=timezone.utc,
        comfort_min_temperature=19.5,
        comfort_max_temperature=20.5,
    )

    assert result.horizon_hours == 24
    assert result.total_days == 1
    assert result.successful_days == 1
    assert result.failed_days == 0
    assert result.average_rmse == pytest.approx(0.0)
    assert result.average_bias == pytest.approx(0.0)
    assert result.average_max_absolute_error == pytest.approx(0.0)
    assert result.worst_day_by_rmse == date(2026, 4, 28)
    assert result.day_results[0].horizon_hours == 24
    assert result.day_results[0].overlap_count == 95
    assert result.day_results[0].minimum_predicted_temperature == pytest.approx(20.0)
    assert result.day_results[0].maximum_predicted_temperature == pytest.approx(20.0)
    assert result.day_results[0].under_comfort_count == 0
    assert result.day_results[0].over_comfort_count == 0
    assert result.day_results[0].error is None


def test_prediction_service_backtests_configured_horizon_hours() -> None:
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
            "previous_room_temperature": 1.0,
            "outdoor_temperature": 0.0,
            "previous_thermostat_setpoint": 0.0,
            "gti_living_room_windows_adjusted": 0.0,
        },
        intercept=0.0,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )
    service = RoomTemperaturePredictionService(
        FakeBacktestReader(),
        FakeModelRepository(model),
    )
    backtesting_service = RoomTemperatureBacktestingService(
        FakeBacktestReader(),
        FakeModelRepository(model),
        service,
    )

    result = backtesting_service.backtest_by_day(
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 28),
        horizon_hours=6,
        timezone_info=timezone.utc,
    )

    assert result.horizon_hours == 6
    assert result.day_results[0].horizon_hours == 6
    assert result.day_results[0].overlap_count == 23
