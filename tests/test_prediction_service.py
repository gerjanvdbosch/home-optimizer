from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TARGET_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    IdentifiedModel,
    NumericPoint,
    NumericSeries,
    ShutterPositionControl,
    ThermostatSetpointControl,
    TextPoint,
    TextSeries,
)
from home_optimizer.features.prediction.schemas import RoomTemperatureControlInputs
from home_optimizer.features.prediction import RoomTemperaturePredictionService


class FakePredictionReader:
    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        series_by_name = {
            "room_temperature": NumericSeries(
                name="room_temperature",
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T09:45:00+00:00", value=20.0),
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=20.5),
                ],
            ),
            HP_FLOW: NumericSeries(
                name=HP_FLOW,
                unit="Lmin",
                points=[
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=12.0),
                    NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=12.0),
                    NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=12.0),
                ],
            ),
            HP_SUPPLY_TEMPERATURE: NumericSeries(
                name=HP_SUPPLY_TEMPERATURE,
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=32.0),
                    NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=32.0),
                    NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=32.0),
                ],
            ),
            HP_RETURN_TEMPERATURE: NumericSeries(
                name=HP_RETURN_TEMPERATURE,
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=30.0),
                    NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=30.0),
                    NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=30.0),
                ],
            ),
            HP_SUPPLY_TARGET_TEMPERATURE: NumericSeries(
                name=HP_SUPPLY_TARGET_TEMPERATURE,
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=31.0),
                    NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=31.0),
                    NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=31.0),
                ],
            ),
            DEFROST_ACTIVE: NumericSeries(
                name=DEFROST_ACTIVE,
                unit="bool",
                points=[],
            ),
            BOOSTER_HEATER_ACTIVE: NumericSeries(
                name=BOOSTER_HEATER_ACTIVE,
                unit="bool",
                points=[],
            ),
        }
        return [series_by_name[name] for name in names]

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

    def read_text_series(self, names, start_time, end_time) -> list[TextSeries]:
        series_by_name = {
            HP_MODE: TextSeries(
                name=HP_MODE,
                points=[
                    TextPoint(timestamp="2026-04-28T10:00:00+00:00", value="heat"),
                ],
            )
        }
        return [series_by_name[name] for name in names]


class FakeModelRepository:
    def __init__(self, model: IdentifiedModel | list[IdentifiedModel]) -> None:
        self.models = model if isinstance(model, list) else [model]

    def latest(self, *, model_kind: str, model_name: str | None = None) -> IdentifiedModel | None:
        matches = [item for item in self.models if item.model_kind == model_kind]
        if model_name is not None:
            matches = [item for item in matches if item.model_name == model_name]
        return matches[-1] if matches else None


def test_prediction_service_simulates_multiple_steps() -> None:
    model = IdentifiedModel(
        model_kind="room_temperature",
        model_name="linear_2state_room_temperature",
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
            "gti_living_room_windows_adjusted": 0.001,
            "floor_heat_state": 0.5,
        },
        intercept=0.5,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )
    service = RoomTemperaturePredictionService(
        FakePredictionReader(),
        FakeModelRepository(model),
    )

    thermostat_schedule = NumericSeries(
        name="thermostat_setpoint",
        unit="degC",
        points=[
            NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=21.0),
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
        control_inputs=RoomTemperatureControlInputs(
            thermostat_setpoint=ThermostatSetpointControl.from_schedule(thermostat_schedule),
            shutter_position=ShutterPositionControl.from_schedule(shutter_schedule),
        ),
    )

    assert prediction.model_name == "linear_2state_room_temperature"
    assert [point.timestamp for point in prediction.room_temperature.points] == [
        "2026-04-28T10:15:00+00:00",
        "2026-04-28T10:30:00+00:00",
        "2026-04-28T10:45:00+00:00",
        "2026-04-28T11:00:00+00:00",
    ]
    assert [point.value for point in prediction.room_temperature.points] == pytest.approx(
        [19.225116, 18.0777204, 17.04506436, 16.115673924000003]
    )


def test_prediction_service_returns_prediction_vs_actual() -> None:
    model = IdentifiedModel(
        model_kind="room_temperature",
        model_name="linear_2state_room_temperature",
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
            "gti_living_room_windows_adjusted": 0.001,
            "floor_heat_state": 0.5,
        },
        intercept=0.5,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )
    service = RoomTemperaturePredictionService(
        FakePredictionReader(),
        FakeModelRepository(model),
    )

    thermostat_schedule = NumericSeries(
        name="thermostat_setpoint",
        unit="degC",
        points=[
            NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=21.0),
        ],
    )

    comparison = service.predict_vs_actual(
        start_time=datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 28, 10, 30, tzinfo=timezone.utc),
        control_inputs=RoomTemperatureControlInputs(
            thermostat_setpoint=ThermostatSetpointControl.from_schedule(thermostat_schedule),
        ),
    )

    assert comparison.model_name == "linear_2state_room_temperature"
    assert [point.timestamp for point in comparison.predicted_room_temperature.points] == [
        "2026-04-28T10:15:00+00:00",
        "2026-04-28T10:30:00+00:00",
    ]
    assert [point.timestamp for point in comparison.actual_room_temperature.points] == [
        "2026-04-28T10:15:00+00:00",
        "2026-04-28T10:30:00+00:00",
    ]


def test_prediction_service_uses_thermal_output_response_model_when_available() -> None:
    room_model = IdentifiedModel(
        model_kind="room_temperature",
        model_name="linear_2state_room_temperature",
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
            "gti_living_room_windows_adjusted": 0.001,
            "floor_heat_state": 0.5,
        },
        intercept=0.5,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="room_temperature",
    )
    thermal_model = IdentifiedModel(
        model_kind="thermal_output",
        model_name="linear_1step_thermal_output",
        trained_at_utc=datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc),
        training_start_time_utc=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        training_end_time_utc=datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        sample_count=100,
        train_sample_count=80,
        test_sample_count=20,
        coefficients={
            "previous_thermal_output": 0.8,
            "previous_heating_demand": 0.6,
            "previous_floor_heat_state": 0.4,
            "outdoor_temperature": -0.05,
            "hp_supply_target_temperature": 0.1,
        },
        intercept=0.3,
        train_rmse=0.05,
        test_rmse=0.1,
        test_rmse_recursive=0.14,
        target_name="thermal_output",
    )
    service = RoomTemperaturePredictionService(
        FakePredictionReader(),
        FakeModelRepository([room_model, thermal_model]),
    )

    thermostat_schedule = NumericSeries(
        name="thermostat_setpoint",
        unit="degC",
        points=[
            NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=21.0),
            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=21.0),
        ],
    )

    prediction = service.predict(
        start_time=datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 28, 10, 30, tzinfo=timezone.utc),
        control_inputs=RoomTemperatureControlInputs(
            thermostat_setpoint=ThermostatSetpointControl.from_schedule(thermostat_schedule),
        ),
    )

    assert prediction.model_name == "linear_2state_room_temperature"
    assert len(prediction.room_temperature.points) == 2
