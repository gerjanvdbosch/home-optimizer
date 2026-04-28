from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.domain import NumericPoint, NumericSeries, TextPoint, TextSeries
from home_optimizer.domain import BuildingTemperatureModel
from home_optimizer.features.identification import BuildingModelIdentificationService


class FakeIdentificationReader:
    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        base_time = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        room_points: list[NumericPoint] = []
        outdoor_points: list[NumericPoint] = []
        setpoint_points: list[NumericPoint] = []
        shutter_points: list[NumericPoint] = []
        hp_power_points: list[NumericPoint] = []

        room_temperature = 20.0
        for index in range(48):
            timestamp = (base_time + timedelta(minutes=5 * index)).isoformat()
            outdoor = 8.0 + 0.05 * index
            setpoint = 21.0 if index < 24 else 20.5
            hp_power = 0.8 + 0.01 * (index % 5)
            solar_gain = 120.0 if 12 <= index <= 30 else 0.0

            room_temperature = (
                0.9 * room_temperature
                + 0.03 * outdoor
                + 0.02 * setpoint
                + 0.015 * hp_power
                + 0.0008 * solar_gain
                + 0.35
            )

            room_points.append(NumericPoint(timestamp=timestamp, value=room_temperature))
            outdoor_points.append(NumericPoint(timestamp=timestamp, value=outdoor))
            setpoint_points.append(NumericPoint(timestamp=timestamp, value=setpoint))
            shutter_points.append(NumericPoint(timestamp=timestamp, value=100.0))
            hp_power_points.append(NumericPoint(timestamp=timestamp, value=hp_power))

        return [
            NumericSeries(name="room_temperature", unit="degC", points=room_points),
            NumericSeries(name="outdoor_temperature", unit="degC", points=outdoor_points),
            NumericSeries(name="thermostat_setpoint", unit="degC", points=setpoint_points),
            NumericSeries(name="shutter_living_room", unit="percent", points=shutter_points),
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        base_time = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        points = []
        for index in range(48):
            timestamp = (base_time + timedelta(minutes=5 * index)).isoformat()
            solar_gain = 120.0 if 12 <= index <= 30 else 0.0
            points.append(NumericPoint(timestamp=timestamp, value=solar_gain))
        return [NumericSeries(name="gti_living_room_windows", unit="Wm2", points=points)]

    def read_text_series(self, names, start_time, end_time) -> list[TextSeries]:
        return [
            TextSeries(
                name="hp_mode",
                points=[TextPoint(timestamp="2026-04-25T00:00:00+00:00", value="heat")],
            )
        ]


class FakeBuildingModelRepository:
    def __init__(self) -> None:
        self.saved: list[BuildingTemperatureModel] = []

    def save(self, model: BuildingTemperatureModel) -> None:
        self.saved.append(model)

    def latest(self) -> BuildingTemperatureModel | None:
        return self.saved[-1] if self.saved else None


def test_identify_and_store_persists_model() -> None:
    repository = FakeBuildingModelRepository()
    service = BuildingModelIdentificationService(
        FakeIdentificationReader(),
        model_repository=repository,
    )

    model = service.identify_and_store(
        start_time=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
        interval_minutes=15,
    )

    assert repository.saved == [model]
    assert model.model_name == "linear_1step_room_temperature"
    assert model.interval_minutes == 15
    assert model.target_name == "room_temperature"
