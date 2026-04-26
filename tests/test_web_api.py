from __future__ import annotations

from datetime import date, datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.charts import ChartPoint, ChartSeries, ChartTextPoint, ChartTextSeries
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.features.history_import.schemas import HistoryImportResult
from home_optimizer.web.app import create_app
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


class FakeDashboardRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str, str]] = []

    def read_series(self, names, start_time, end_time) -> list[ChartSeries]:
        self.calls.append(("numeric", names, start_time.isoformat(), end_time.isoformat()))
        if names == ["shutter_living_room"]:
            return [
                ChartSeries(
                    name="shutter_living_room",
                    unit="percent",
                    points=[ChartPoint(timestamp="2026-04-25T11:55:00+00:00", value=50.0)],
                )
            ]
        return [
            ChartSeries(
                name="room_temperature",
                unit="degC",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=20.5)],
            ),
            ChartSeries(
                name="thermostat_setpoint",
                unit="degC",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=21.0)],
            ),
            ChartSeries(
                name="dhw_top_temperature",
                unit="degC",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=48.0)],
            ),
            ChartSeries(
                name="dhw_bottom_temperature",
                unit="degC",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=42.0)],
            ),
            ChartSeries(
                name="hp_electric_power",
                unit="W",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=1500.0)],
            ),
            ChartSeries(
                name="defrost_active",
                unit="bool",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=0.0)],
            ),
            ChartSeries(
                name="booster_heater_active",
                unit="bool",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=1.0)],
            ),
        ]

    def read_text_series(self, names, start_time, end_time) -> list[ChartTextSeries]:
        self.calls.append(("text", names, start_time.isoformat(), end_time.isoformat()))
        return [
            ChartTextSeries(
                name="hp_mode",
                points=[ChartTextPoint(timestamp="2026-04-25T11:50:00+00:00", value="ufh")],
            ),
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[ChartSeries]:
        self.calls.append(("forecast", names, start_time.isoformat(), end_time.isoformat()))
        return [
            ChartSeries(
                name="temperature",
                unit="degC",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=12.5)],
            ),
            ChartSeries(
                name="gti_pv",
                unit="Wm2",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=500.0)],
            ),
            ChartSeries(
                name="gti_living_room_windows",
                unit="Wm2",
                points=[ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=220.0)],
            ),
        ]


class FakeContainer:
    def __init__(
        self,
        history_import_service: FakeHistoryImportService,
        home_assistant: FakeHomeAssistantGateway,
    ) -> None:
        self.history_import_service = history_import_service
        self.home_assistant = home_assistant
        self.dashboard_repository = FakeDashboardRepository()
        self.telemetry_scheduler = FakeTelemetryScheduler()
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
    assert 'href="static/app.css"' in response.text
    assert 'src="plotly.js"' in response.text
    assert 'src="static/app.js"' in response.text
    assert 'href="/static/app.css"' not in response.text
    assert "sensor.room_temperature" not in response.text
    assert app.state.container.telemetry_scheduler.started is True
    assert app.state.container.forecast_scheduler.started is True
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
        "points": [{"timestamp": "2026-04-25T11:50:00+00:00", "value": "ufh"}],
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
    local_timezone = dashboard_charts_module.current_timezone()
    start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
    end_time = start_time + timedelta(days=1)
    assert app.state.container.dashboard_repository.calls == [
        (
            "numeric",
            ["shutter_living_room"],
            (start_time - timedelta(days=1)).isoformat(),
            end_time.isoformat(),
        ),
        (
            "numeric",
            [
                "room_temperature",
                "thermostat_setpoint",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
                "defrost_active",
                "booster_heater_active",
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
    assert app.state.container.dashboard_repository.calls[1][2:] == (
        "2026-04-25T00:00:00+02:00",
        "2026-04-26T00:00:00+02:00",
    )


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
            client.get("/static/app.js"),
            client.get("/static/app.css"),
            client.get("/plotly.js"),
        ]

    for response in responses:
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"
