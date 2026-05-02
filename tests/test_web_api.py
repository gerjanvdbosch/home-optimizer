from __future__ import annotations

from datetime import date, datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    IdentifiedModel,
    NumericPoint,
    NumericSeries,
    ShutterPositionControl,
    ThermostatSetpointControl,
    TextPoint,
    TextSeries,
    build_sensor_specs,
)
from home_optimizer.features import (
    HistoryImportResult,
    IdentificationResult,
    RoomTemperatureControlInputs,
    RoomTemperaturePrediction,
    RoomTemperaturePredictionComparison,
    ThermostatSetpointCandidateEvaluation,
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointMpcPlanRequest,
)
from home_optimizer.web import create_app
from home_optimizer.web.services import dashboard_charts as dashboard_charts_module


class FakeHomeAssistantGateway:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeHistoryImportService:
    def __init__(self, result: HistoryImportResult) -> None:
        self.result = result
        self.calls = 0

    def import_many(self, request) -> HistoryImportResult:
        self.calls += 1
        return self.result


class FakeTelemetryScheduler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeWeatherImportService:
    def __init__(self) -> None:
        self.import_calls = 0

    def import_weather_data(
        self,
        created_at: datetime | None = None,
    ) -> int:
        self.import_calls += 1
        return 12


class FakeHistoricalWeatherImportService:
    def __init__(self) -> None:
        self.import_calls = 0

    def import_historical_weather(
        self,
        created_at: datetime | None = None,
    ) -> int:
        self.import_calls += 1
        return 24


class FakeIdentificationService:
    def __init__(self, result: IdentificationResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str, int, float]] = []

    def identify(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        self.calls.append(
            (
                start_time.isoformat(),
                end_time.isoformat(),
                interval_minutes,
                train_fraction,
            )
        )
        return self.result


class FakeModelTrainingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, float]] = []

    def train_all_models(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> list[IdentifiedModel]:
        self.calls.append(
            (
                start_time.isoformat(),
                end_time.isoformat(),
                interval_minutes,
                train_fraction,
            )
        )
        return [
            IdentifiedModel(
                model_kind="thermal_output",
                model_name="linear_1step_thermal_output",
                trained_at_utc=datetime(2026, 4, 28, 18, 0),
                training_start_time_utc=start_time,
                training_end_time_utc=end_time,
                interval_minutes=interval_minutes,
                sample_count=168,
                train_sample_count=134,
                test_sample_count=34,
                coefficients={
                    "previous_thermal_output": 0.7,
                    "previous_heating_demand": 0.2,
                    "previous_floor_heat_state": 0.15,
                    "outdoor_temperature": -0.01,
                    "hp_supply_target_temperature": 0.08,
                },
                intercept=0.05,
                train_rmse=0.11,
                test_rmse=0.18,
                test_rmse_recursive=0.22,
                target_name="thermal_output",
            ),
            IdentifiedModel(
                model_kind="room_temperature",
                model_name="linear_2state_room_temperature",
                trained_at_utc=datetime(2026, 4, 28, 18, 1),
                training_start_time_utc=start_time,
                training_end_time_utc=end_time,
                interval_minutes=interval_minutes,
                sample_count=168,
                train_sample_count=134,
                test_sample_count=34,
                coefficients={
                    "previous_room_temperature": 0.94,
                    "outdoor_temperature": 0.01,
                    "floor_heat_state": 0.06,
                    "gti_living_room_windows_adjusted": 0.0003,
                },
                intercept=0.02,
                train_rmse=0.06,
                test_rmse=0.13,
                test_rmse_recursive=0.21,
                target_name="room_temperature",
            ),
        ]


class FakePredictionService:
    def __init__(self, result: RoomTemperaturePrediction) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []
        self.comparison_calls: list[dict[str, object]] = []

    def predict(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        control_inputs: RoomTemperatureControlInputs,
        model_name: str = "linear_2state_room_temperature",
    ) -> RoomTemperaturePrediction:
        self.calls.append(
            {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "control_inputs": control_inputs,
                "model_name": model_name,
            }
        )
        return self.result

    def predict_vs_actual(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        control_inputs: RoomTemperatureControlInputs,
        model_name: str = "linear_2state_room_temperature",
    ) -> RoomTemperaturePredictionComparison:
        self.comparison_calls.append(
            {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "control_inputs": control_inputs,
                "model_name": model_name,
            }
        )
        return RoomTemperaturePredictionComparison(
            model_name=self.result.model_name,
            interval_minutes=self.result.interval_minutes,
            target_name=self.result.target_name,
            predicted_room_temperature=self.result.room_temperature,
            actual_room_temperature=NumericSeries(
                name="room_temperature",
                unit="degC",
                points=[
                    NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=20.6),
                    NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=20.8),
                ],
            ),
        )


class FakeMpcPlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose_plan(
        self,
        request: ThermostatSetpointMpcPlanRequest,
        *,
        shutter_position: ShutterPositionControl | None = None,
    ) -> ThermostatSetpointMpcEvaluationResult:
        self.calls.append(
            {
                "request": request,
                "shutter_position": shutter_position,
            }
        )
        best_schedule = NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=19.0),
                NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=19.0),
                NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=20.0),
            ],
        )
        best_prediction = NumericSeries(
            name="room_temperature",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=19.4),
                NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=19.7),
            ],
        )
        runner_up_schedule = NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=20.0),
                NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=20.0),
                NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=20.0),
            ],
        )
        runner_up_prediction = NumericSeries(
            name="room_temperature",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=20.2),
                NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=20.5),
            ],
        )
        best_candidate = ThermostatSetpointCandidateEvaluation(
            candidate_name="candidate_1",
            thermostat_setpoint_schedule=best_schedule,
            predicted_room_temperature=best_prediction,
            total_cost=0.25,
            comfort_violation_cost=0.0,
            setpoint_change_cost=0.25,
            minimum_predicted_temperature=19.4,
            maximum_predicted_temperature=19.7,
        )
        return ThermostatSetpointMpcEvaluationResult(
            model_name="linear_2state_room_temperature",
            interval_minutes=request.interval_minutes,
            candidate_results=[
                best_candidate,
                ThermostatSetpointCandidateEvaluation(
                    candidate_name="candidate_2",
                    thermostat_setpoint_schedule=runner_up_schedule,
                    predicted_room_temperature=runner_up_prediction,
                    total_cost=0.6,
                    comfort_violation_cost=0.5,
                    setpoint_change_cost=0.1,
                    minimum_predicted_temperature=20.2,
                    maximum_predicted_temperature=20.5,
                ),
            ],
            best_candidate=best_candidate,
        )


class FakeTimeSeriesReadRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str, str]] = []

    def sample_time_range(self) -> tuple[datetime | None, datetime | None]:
        return (
            datetime(2026, 4, 20, 0, 0),
            datetime(2026, 4, 28, 23, 59),
        )

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        self.calls.append(("numeric", names, start_time.isoformat(), end_time.isoformat()))
        if names == ["shutter_living_room"]:
            return [
                NumericSeries(
                    name="shutter_living_room",
                    unit="percent",
                    points=[NumericPoint(timestamp="2026-04-25T11:55:00+00:00", value=50.0)],
                )
            ]
        return [
            NumericSeries(
                name="room_temperature",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=20.5)],
            ),
            NumericSeries(
                name="thermostat_setpoint",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=21.0)],
            ),
            NumericSeries(
                name="dhw_top_temperature",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=48.0)],
            ),
            NumericSeries(
                name="dhw_bottom_temperature",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=42.0)],
            ),
            NumericSeries(
                name="hp_electric_power",
                unit="W",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=1500.0)],
            ),
            NumericSeries(
                name="defrost_active",
                unit="bool",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=0.0)],
            ),
            NumericSeries(
                name="booster_heater_active",
                unit="bool",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=1.0)],
            ),
        ]

    def read_text_series(self, names, start_time, end_time) -> list[TextSeries]:
        self.calls.append(("text", names, start_time.isoformat(), end_time.isoformat()))
        return [
            TextSeries(
                name="hp_mode",
                points=[TextPoint(timestamp="2026-04-25T11:50:00+00:00", value="heat")],
            ),
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        self.calls.append(("forecast", names, start_time.isoformat(), end_time.isoformat()))
        return [
            NumericSeries(
                name="temperature",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=12.5)],
            ),
            NumericSeries(
                name="gti_pv",
                unit="Wm2",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=500.0)],
            ),
            NumericSeries(
                name="gti_living_room_windows",
                unit="Wm2",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=220.0)],
            ),
        ]

    def read_historical_weather_series(self, names, start_time, end_time) -> list[NumericSeries]:
        self.calls.append(
            ("historical_weather", names, start_time.isoformat(), end_time.isoformat())
        )
        return [
            NumericSeries(
                name="gti_living_room_windows",
                unit="Wm2",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=210.0)],
            )
        ]


class FakeContainer:
    def __init__(
        self,
        history_import_service: FakeHistoryImportService,
        home_assistant: FakeHomeAssistantGateway,
    ) -> None:
        self.history_import_service = history_import_service
        self.home_assistant = home_assistant
        self.time_series_read_repository = FakeTimeSeriesReadRepository()
        self.identification_service = FakeIdentificationService(
            IdentificationResult(
                model_name="linear_2state_room_temperature",
                interval_minutes=15,
                sample_count=168,
                train_sample_count=134,
                test_sample_count=34,
                coefficients={
                    "previous_room_temperature": 0.94,
                    "outdoor_temperature": 0.01,
                    "floor_heat_state": 0.06,
                    "gti_living_room_windows_adjusted": 0.0003,
                },
                intercept=0.02,
                train_rmse=0.06,
                test_rmse=0.13,
                test_rmse_recursive=0.21,
                target_name="room_temperature",
            )
        )
        self.model_training_service = FakeModelTrainingService()
        self.prediction_service = FakePredictionService(
            RoomTemperaturePrediction(
                model_name="linear_2state_room_temperature",
                interval_minutes=15,
                target_name="room_temperature",
                room_temperature=NumericSeries(
                    name="room_temperature",
                    unit="degC",
                    points=[
                        NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=20.7),
                        NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=20.9),
                    ],
                ),
            )
        )
        self.mpc_planner = FakeMpcPlanner()
        self.weather_import_service = FakeWeatherImportService()
        self.historical_weather_import_service = FakeHistoricalWeatherImportService()
        self.telemetry_scheduler = FakeTelemetryScheduler()
        self.historical_weather_scheduler = FakeTelemetryScheduler()
        self.model_training_scheduler = FakeTelemetryScheduler()
        self.forecast_scheduler = FakeTelemetryScheduler()

    def close(self) -> None:
        self.home_assistant.close()


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(20):
        response = client.get(f"/api/history-import/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        sleep(0.05)

    raise AssertionError("history import job did not finish")


def test_dashboard_shows_import_button() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={"room_temperature": 3}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 14,
            "sensors": {"room_temperature": "sensor.room_temperature"},
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Importeer geschiedenis" in response.text
    assert "Importeer weerdata" in response.text
    assert "Scenario voorspelling" not in response.text
    assert 'href="static/shared.css"' in response.text
    assert 'href="static/dashboard.css"' in response.text
    assert 'src="plotly.js"' in response.text
    assert 'src="static/shared.js"' in response.text
    assert 'src="static/dashboard.js"' in response.text
    assert 'href="/static/shared.css"' not in response.text
    assert 'href="./"' in response.text
    assert 'href="simulation"' in response.text
    assert "sensor.room_temperature" not in response.text
    assert app.state.container.telemetry_scheduler.started is True
    assert app.state.container.historical_weather_scheduler.started is True
    assert app.state.container.model_training_scheduler.started is True
    assert app.state.container.forecast_scheduler.started is True
    assert gateway.closed is True


def test_weather_import_endpoint_runs_forecast_backfill() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={"room_temperature": 3}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 14,
            "sensors": {"room_temperature": "sensor.room_temperature"},
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/api/weather-import")

    assert response.status_code == 200
    assert response.json() == {"imported_rows": 12}
    assert app.state.container.weather_import_service.import_calls == 1


def test_simulation_page_shows_prediction_panel() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={"room_temperature": 3}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 14,
            "sensors": {"room_temperature": "sensor.room_temperature"},
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/simulation")

    assert response.status_code == 200
    assert "Scenario voorspelling vs gemeten" in response.text
    assert "MPC voorstel" in response.text
    assert "MPC top kandidaten" in response.text
    assert "Train alle modellen" in response.text
    assert "thermal-output responsmodel" in response.text
    assert "De gemeten setpointreeks van de startdag wordt gebruikt voor de vergelijking." in response.text
    assert "De gemeten shutterreeks van de startdag wordt gebruikt voor de vergelijking." in response.text
    assert 'href="static/shared.css"' in response.text
    assert 'href="static/simulation.css"' in response.text
    assert 'src="static/shared.js"' in response.text
    assert 'src="static/simulation.js"' in response.text
    assert 'href="./"' in response.text
    assert gateway.closed is True


def test_settings_reject_legacy_sensor_fields() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_options(
            {
                "database_path": "/tmp/home-optimizer-test.db",
                "sensor_room_temperature": "sensor.room_temperature",
            }
        )


def test_settings_require_database_path() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_options({"sensors": {"room_temperature": "sensor.room_temperature"}})


def test_sensor_bindings_can_be_configured_as_mapping() -> None:
    settings = AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "sensors": {"room_temperature": " sensor.room_temperature "},
        }
    )

    specs = build_sensor_specs(settings)

    assert [spec.name for spec in specs] == ["room_temperature"]
    assert specs[0].entity_id == "sensor.room_temperature"


def test_sensor_bindings_reject_object_form() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_options(
            {
                "database_path": "/tmp/home-optimizer-test.db",
                "sensors": {"room_temperature": {"entity_id": "sensor.room_temperature"}},
            }
        )


def test_history_import_endpoint_returns_summary() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(
        HistoryImportResult(imported_rows={"room_temperature": 3, "outdoor_temperature": 7})
    )
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 10,
            "sensors": {
                "room_temperature": "sensor.room_temperature",
                "outdoor_temperature": "sensor.outdoor_temperature",
            },
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")
        job = wait_for_job(client, response.json()["job_id"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"pending", "running"}
    assert payload["sensor_count"] == 2
    assert payload["job_id"]
    assert job["status"] == "succeeded"
    assert service.calls == 1
    assert gateway.closed is True


def test_history_import_job_endpoint_returns_result() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(
        HistoryImportResult(imported_rows={"room_temperature": 3, "outdoor_temperature": 7})
    )
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 10,
            "sensors": {
                "room_temperature": "sensor.room_temperature",
                "outdoor_temperature": "sensor.outdoor_temperature",
            },
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")
        assert response.status_code == 200
        payload = wait_for_job(client, response.json()["job_id"])

    assert payload["status"] == "succeeded"
    assert payload["imported_rows"] == {
        "room_temperature": 3,
        "outdoor_temperature": 7,
    }
    assert payload["total_rows"] == 10
    assert payload["sensor_count"] == 2
    assert payload["error"] is None


def test_dashboard_charts_endpoint_returns_day_series() -> None:
    chart_date = date(2026, 4, 25)
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/dashboard/charts?date=2026-04-25")

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "2026-04-25"
    assert payload["room_temperature"]["points"] == [
        {"timestamp": "2026-04-25T12:00:00+00:00", "value": 20.5}
    ]
    assert payload["thermostat_setpoint"] == {
        "name": "thermostat_setpoint",
        "unit": "degC",
        "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 21.0}],
    }
    assert [series["name"] for series in payload["dhw_temperatures"]] == [
        "dhw_top_temperature",
        "dhw_bottom_temperature",
    ]
    assert payload["heatpump_power"] == {
        "name": "hp_electric_power",
        "unit": "W",
        "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 1500.0}],
    }
    assert payload["heatpump_mode"] == {
        "name": "hp_mode",
        "points": [{"timestamp": "2026-04-25T11:50:00+00:00", "value": "heat"}],
    }
    assert payload["heatpump_statuses"] == [
        {
            "name": "defrost_active",
            "unit": "bool",
            "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 0.0}],
        },
        {
            "name": "booster_heater_active",
            "unit": "bool",
            "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 1.0}],
        },
    ]
    assert payload["forecast_temperature"] == {
        "name": "temperature",
        "unit": "degC",
        "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 12.5}],
    }
    assert payload["forecast_gti"] == [
        {
            "name": "gti_pv",
            "unit": "Wm2",
            "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 500.0}],
        },
        {
            "name": "gti_living_room_windows",
            "unit": "Wm2",
            "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 220.0}],
        },
        {
            "name": "gti_living_room_windows_adjusted",
            "unit": "Wm2",
            "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 110.0}],
        },
    ]
    assert payload["historical_weather_temperature"] == {
        "name": "temperature",
        "unit": None,
        "points": [],
    }
    assert payload["historical_weather_gti"][0] == {
        "name": "gti_pv",
        "unit": None,
        "points": [],
    }
    assert payload["historical_weather_gti"][1]["name"] == "gti_living_room_windows"
    assert payload["historical_weather_gti"][1]["unit"] == "Wm2"
    assert payload["historical_weather_gti"][1]["points"][:3] == [
        {"timestamp": "2026-04-25T12:00:00+00:00", "value": 210.0},
        {"timestamp": "2026-04-25T12:15:00+00:00", "value": 210.0},
        {"timestamp": "2026-04-25T12:30:00+00:00", "value": 210.0},
    ]
    assert payload["historical_weather_gti"][2]["name"] == "gti_living_room_windows_adjusted"
    assert payload["historical_weather_gti"][2]["unit"] == "Wm2"
    assert payload["historical_weather_gti"][2]["points"][:3] == [
        {"timestamp": "2026-04-25T12:00:00+00:00", "value": 105.0},
        {"timestamp": "2026-04-25T12:15:00+00:00", "value": 105.0},
        {"timestamp": "2026-04-25T12:30:00+00:00", "value": 105.0},
    ]
    local_timezone = dashboard_charts_module.current_timezone()
    start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
    end_time = start_time + timedelta(days=1)
    forecast_end_time = end_time + timedelta(minutes=15)
    assert app.state.container.time_series_read_repository.calls == [
        (
            "numeric",
            ["shutter_living_room"],
            start_time.isoformat(),
            end_time.isoformat(),
        ),
        (
            "numeric",
            [
                "room_temperature",
                "thermostat_setpoint",
                "hp_flow",
                "p1_net_power",
                "pv_output_power",
                "hp_supply_temperature",
                "hp_supply_target_temperature",
                "hp_return_temperature",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
                "defrost_active",
                "booster_heater_active",
                "compressor_frequency",
            ],
            start_time.isoformat(),
            end_time.isoformat(),
        ),
        (
            "text",
            ["hp_mode"],
            start_time.isoformat(),
            end_time.isoformat(),
        ),
        (
            "forecast",
            ["temperature", "gti_pv", "gti_living_room_windows"],
            start_time.isoformat(),
            forecast_end_time.isoformat(),
        ),
        (
            "historical_weather",
            ["temperature", "gti_pv", "gti_living_room_windows"],
            start_time.isoformat(),
            end_time.isoformat(),
        ),
    ]


def test_dashboard_charts_endpoint_uses_current_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )
    monkeypatch.setattr(
        dashboard_charts_module,
        "current_timezone",
        lambda: ZoneInfo("Europe/Amsterdam"),
    )

    with TestClient(app) as client:
        response = client.get("/api/dashboard/charts?date=2026-04-25")

    assert response.status_code == 200
    assert app.state.container.time_series_read_repository.calls[1][2:] == (
        "2026-04-25T00:00:00+02:00",
        "2026-04-26T00:00:00+02:00",
    )


def test_identification_endpoint_returns_model_fit() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/identification",
            params={
                "start_time": "2026-04-25T19:15:00+00:00",
                "end_time": "2026-04-28T17:00:00+00:00",
                "interval_minutes": 30,
                "train_fraction": 0.75,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "model_name": "linear_2state_room_temperature",
        "interval_minutes": 15,
        "sample_count": 168,
        "train_sample_count": 134,
        "test_sample_count": 34,
        "coefficients": {
            "previous_room_temperature": 0.94,
            "outdoor_temperature": 0.01,
            "floor_heat_state": 0.06,
            "gti_living_room_windows_adjusted": 0.0003,
        },
        "intercept": 0.02,
        "train_rmse": 0.06,
        "test_rmse": 0.13,
        "test_rmse_recursive": 0.21,
        "target_name": "room_temperature",
    }
    assert app.state.container.identification_service.calls == [
        (
            "2026-04-25T19:15:00+00:00",
            "2026-04-28T17:00:00+00:00",
            30,
            0.75,
        )
    ]


def test_identification_train_endpoint_stores_all_models() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/identification/train",
            json={
                "start_time": "2026-04-25T00:00:00+00:00",
                "end_time": "2026-04-28T00:00:00+00:00",
                "interval_minutes": 15,
                "train_fraction": 0.8,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert [model["model_name"] for model in payload["models"]] == [
        "linear_1step_thermal_output",
        "linear_2state_room_temperature",
    ]
    assert [model["target_name"] for model in payload["models"]] == [
        "thermal_output",
        "room_temperature",
    ]
    assert app.state.container.model_training_service.calls == [
        (
            "2026-04-25T00:00:00+00:00",
            "2026-04-28T00:00:00+00:00",
            15,
            0.8,
        )
    ]


def test_identification_train_all_endpoint_is_removed() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/identification/train-all",
            json={
                "start_time": "2026-04-25T00:00:00+00:00",
                "end_time": "2026-04-28T00:00:00+00:00",
                "interval_minutes": 15,
                "train_fraction": 0.8,
            },
        )

    assert response.status_code == 404
    assert app.state.container.model_training_service.calls == []


def test_prediction_endpoint_returns_room_temperature_series() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/prediction",
            json={
                "start_time": "2026-04-28T10:00:00+00:00",
                "end_time": "2026-04-28T10:30:00+00:00",
                "thermostat_schedule": {
                    "name": "thermostat_setpoint",
                    "unit": "degC",
                    "points": [
                        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 21.0},
                        {"timestamp": "2026-04-28T10:30:00+00:00", "value": 21.0},
                    ],
                },
                "shutter_schedule": {
                    "name": "shutter_living_room",
                    "unit": "percent",
                    "points": [
                        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 50.0},
                        {"timestamp": "2026-04-28T10:30:00+00:00", "value": 50.0},
                    ],
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "linear_2state_room_temperature",
        "interval_minutes": 15,
        "target_name": "room_temperature",
        "room_temperature": {
            "name": "room_temperature",
            "unit": "degC",
            "points": [
                {"timestamp": "2026-04-28T10:15:00+00:00", "value": 20.7},
                {"timestamp": "2026-04-28T10:30:00+00:00", "value": 20.9},
            ],
        },
    }
    assert app.state.container.prediction_service.calls == [
        {
            "start_time": "2026-04-28T10:00:00+00:00",
            "end_time": "2026-04-28T10:30:00+00:00",
            "control_inputs": RoomTemperatureControlInputs(
                thermostat_setpoint=ThermostatSetpointControl.from_schedule(
                    NumericSeries(
                        name="thermostat_setpoint",
                        unit="degC",
                        points=[
                            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=21.0),
                            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=21.0),
                        ],
                    )
                ),
                shutter_position=ShutterPositionControl.from_schedule(
                    NumericSeries(
                        name="shutter_living_room",
                        unit="percent",
                        points=[
                            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=50.0),
                            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=50.0),
                        ],
                    )
                ),
            ),
            "model_name": "linear_2state_room_temperature",
        }
    ]


def test_prediction_comparison_endpoint_returns_predicted_and_actual_series() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/prediction/compare",
            json={
                "start_time": "2026-04-28T10:00:00+00:00",
                "end_time": "2026-04-28T10:30:00+00:00",
                "thermostat_schedule": {
                    "name": "thermostat_setpoint",
                    "unit": "degC",
                    "points": [
                        {"timestamp": "2026-04-28T10:00:00+00:00", "value": 21.0},
                        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 21.0},
                        {"timestamp": "2026-04-28T10:30:00+00:00", "value": 21.0},
                    ],
                },
                "shutter_schedule": {
                    "name": "shutter_living_room",
                    "unit": "percent",
                    "points": [
                        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 50.0},
                        {"timestamp": "2026-04-28T10:30:00+00:00", "value": 50.0},
                    ],
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "linear_2state_room_temperature"
    assert payload["interval_minutes"] == 15
    assert payload["target_name"] == "room_temperature"
    assert payload["predicted_room_temperature"] == {
        "name": "room_temperature",
        "unit": "degC",
        "points": [
            {"timestamp": "2026-04-28T10:15:00+00:00", "value": 20.7},
            {"timestamp": "2026-04-28T10:30:00+00:00", "value": 20.9},
        ],
    }
    assert payload["actual_room_temperature"] == {
        "name": "room_temperature",
        "unit": "degC",
        "points": [
            {"timestamp": "2026-04-28T10:15:00+00:00", "value": 20.6},
            {"timestamp": "2026-04-28T10:30:00+00:00", "value": 20.8},
        ],
    }
    assert payload["overlap_count"] == 2
    assert payload["rmse"] == pytest.approx(0.1)
    assert payload["bias"] == pytest.approx(0.1)
    assert payload["max_absolute_error"] == pytest.approx(0.1)
    assert app.state.container.prediction_service.comparison_calls == [
        {
            "start_time": "2026-04-28T10:00:00+00:00",
            "end_time": "2026-04-28T10:30:00+00:00",
            "control_inputs": RoomTemperatureControlInputs(
                thermostat_setpoint=ThermostatSetpointControl.from_schedule(
                    NumericSeries(
                        name="thermostat_setpoint",
                        unit="degC",
                        points=[
                            NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=21.0),
                            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=21.0),
                            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=21.0),
                        ],
                    )
                ),
                shutter_position=ShutterPositionControl.from_schedule(
                    NumericSeries(
                        name="shutter_living_room",
                        unit="percent",
                        points=[
                            NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=50.0),
                            NumericPoint(timestamp="2026-04-28T10:30:00+00:00", value=50.0),
                        ],
                    )
                ),
            ),
            "model_name": "linear_2state_room_temperature",
        }
    ]


def test_mpc_plan_endpoint_returns_ranked_candidates() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/mpc/thermostat-setpoint",
            json={
                "start_time": "2026-04-28T10:00:00+00:00",
                "end_time": "2026-04-28T10:30:00+00:00",
                "interval_minutes": 15,
                "allowed_setpoints": [19.0, 20.0, 21.0],
                "switch_times": ["2026-04-28T10:30:00+00:00"],
                "comfort_min_temperature": 19.0,
                "comfort_max_temperature": 21.0,
                "setpoint_change_penalty": 0.25,
                "shutter_schedule": {
                    "name": "shutter_living_room",
                    "unit": "percent",
                    "points": [
                        {"timestamp": "2026-04-28T10:00:00+00:00", "value": 60.0},
                        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 50.0},
                    ],
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "linear_2state_room_temperature"
    assert payload["interval_minutes"] == 15
    assert payload["best_candidate"]["candidate_name"] == "candidate_1"
    assert payload["best_candidate"]["total_cost"] == 0.25
    assert payload["best_candidate"]["thermostat_setpoint_schedule"]["points"] == [
        {"timestamp": "2026-04-28T10:00:00+00:00", "value": 19.0},
        {"timestamp": "2026-04-28T10:15:00+00:00", "value": 19.0},
        {"timestamp": "2026-04-28T10:30:00+00:00", "value": 20.0},
    ]
    assert payload["candidate_results"][1]["candidate_name"] == "candidate_2"
    assert app.state.container.mpc_planner.calls == [
        {
            "request": ThermostatSetpointMpcPlanRequest(
                start_time=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("UTC")),
                end_time=datetime(2026, 4, 28, 10, 30, tzinfo=ZoneInfo("UTC")),
                interval_minutes=15,
                allowed_setpoints=[19.0, 20.0, 21.0],
                switch_times=[datetime(2026, 4, 28, 10, 30, tzinfo=ZoneInfo("UTC"))],
                comfort_min_temperature=19.0,
                comfort_max_temperature=21.0,
                setpoint_change_penalty=0.25,
            ),
            "shutter_position": ShutterPositionControl.from_schedule(
                NumericSeries(
                    name="shutter_living_room",
                    unit="percent",
                    points=[
                        NumericPoint(timestamp="2026-04-28T10:00:00+00:00", value=60.0),
                        NumericPoint(timestamp="2026-04-28T10:15:00+00:00", value=50.0),
                    ],
                )
            ),
        }
    ]


def test_plotly_script_is_served_locally() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/plotly.js")

    assert response.status_code == 200
    assert "Plotly" in response.text[:5000]


def test_javascript_and_css_are_not_cached() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        responses = [
            client.get("/static/shared.js"),
            client.get("/static/dashboard.js"),
            client.get("/static/simulation.js"),
            client.get("/static/shared.css"),
            client.get("/static/dashboard.css"),
            client.get("/static/simulation.css"),
            client.get("/plotly.js"),
        ]

    for response in responses:
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"
