from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.domain import NumericPoint, NumericSeries
from home_optimizer.features.identification import BuildingModelIdentificationService


class FakeIdentificationReader:
    def __init__(self) -> None:
        self.series_calls: list[tuple[list[str], str, str]] = []
        self.forecast_calls: list[tuple[list[str], str, str]] = []

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        self.series_calls.append((names, start_time.isoformat(), end_time.isoformat()))
        base_time = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        room_points: list[NumericPoint] = []
        outdoor_points: list[NumericPoint] = []
        setpoint_points: list[NumericPoint] = []
        shutter_points: list[NumericPoint] = []
        flow_points: list[NumericPoint] = []
        supply_points: list[NumericPoint] = []
        return_points: list[NumericPoint] = []
        hp_power_points: list[NumericPoint] = []

        room_temperature = 20.0
        for index in range(48):
            timestamp = (base_time + timedelta(minutes=5 * index)).isoformat()
            outdoor = 8.0 + 0.05 * index
            setpoint = 21.0 if index < 24 else 20.5
            hp_power = 0.8 + 0.01 * (index % 5)
            flow = 12.0 + 0.2 * (index % 3)
            supply = 33.0 + 0.03 * index
            return_temperature = 29.0 + 0.01 * index
            solar_gain = 120.0 if 12 <= index <= 30 else 0.0
            thermal_output = flow * max(supply - return_temperature, 0.0) * 4186.0 / 60000.0

            room_temperature = (
                0.9 * room_temperature
                + 0.03 * outdoor
                + 0.02 * setpoint
                + 0.015 * hp_power
                + 0.0008 * solar_gain
                + 0.01 * thermal_output
                + 0.35
            )

            room_points.append(NumericPoint(timestamp=timestamp, value=room_temperature))
            outdoor_points.append(NumericPoint(timestamp=timestamp, value=outdoor))
            setpoint_points.append(NumericPoint(timestamp=timestamp, value=setpoint))
            shutter_points.append(NumericPoint(timestamp=timestamp, value=100.0))
            flow_points.append(NumericPoint(timestamp=timestamp, value=flow))
            supply_points.append(NumericPoint(timestamp=timestamp, value=supply))
            return_points.append(NumericPoint(timestamp=timestamp, value=return_temperature))
            hp_power_points.append(NumericPoint(timestamp=timestamp, value=hp_power))

        return [
            NumericSeries(name="room_temperature", unit="degC", points=room_points),
            NumericSeries(name="outdoor_temperature", unit="degC", points=outdoor_points),
            NumericSeries(name="thermostat_setpoint", unit="degC", points=setpoint_points),
            NumericSeries(name="shutter_living_room", unit="percent", points=shutter_points),
            NumericSeries(name="hp_flow", unit="Lmin", points=flow_points),
            NumericSeries(name="hp_supply_temperature", unit="degC", points=supply_points),
            NumericSeries(name="hp_return_temperature", unit="degC", points=return_points),
            NumericSeries(name="hp_electric_power", unit="kW", points=hp_power_points),
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        self.forecast_calls.append((names, start_time.isoformat(), end_time.isoformat()))
        base_time = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        points = []
        for index in range(48):
            timestamp = (base_time + timedelta(minutes=5 * index)).isoformat()
            solar_gain = 120.0 if 12 <= index <= 30 else 0.0
            points.append(NumericPoint(timestamp=timestamp, value=solar_gain))
        return [NumericSeries(name="gti_living_room_windows", unit="Wm2", points=points)]


def test_build_dataset_resamples_and_returns_feature_matrix() -> None:
    reader = FakeIdentificationReader()
    service = BuildingModelIdentificationService(reader)

    dataset = service.build_dataset(
        start_time=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        interval_minutes=15,
    )

    assert dataset.target_name == "room_temperature"
    assert dataset.feature_names == [
        "previous_room_temperature",
        "outdoor_temperature",
        "thermostat_setpoint",
        "gti_living_room_windows_adjusted",
    ]
    assert len(dataset.timestamps) == len(dataset.features) == len(dataset.targets)
    assert len(dataset.features) >= 10
    assert len(dataset.features[0]) == len(dataset.feature_names)
    assert dataset.features[1][0] == dataset.targets[0]


def test_identify_fits_linear_baseline_and_reports_metrics() -> None:
    reader = FakeIdentificationReader()
    service = BuildingModelIdentificationService(reader)

    result = service.identify(
        start_time=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        interval_minutes=15,
    )

    assert result.model_name == "linear_1step_room_temperature"
    assert result.sample_count == result.train_sample_count + result.test_sample_count
    assert result.train_sample_count > result.test_sample_count >= 1
    assert set(result.coefficients) == {
        "previous_room_temperature",
        "outdoor_temperature",
        "thermostat_setpoint",
        "gti_living_room_windows_adjusted",
    }
    assert result.train_rmse >= 0.0
    assert result.test_rmse >= 0.0
