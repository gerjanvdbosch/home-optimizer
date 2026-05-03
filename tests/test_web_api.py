from __future__ import annotations

from datetime import date, datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    TextPoint,
    TextSeries,
    build_sensor_specs,
)
from home_optimizer.features import HistoryImportResult
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


class FakeScheduler:
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

    def import_weather_data(self, created_at: datetime | None = None) -> int:
        self.import_calls += 1
        return 12


class FakeTimeSeriesReadRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str, str]] = []

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
            )
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
        self.weather_import_service = FakeWeatherImportService()
        self.telemetry_scheduler = FakeScheduler()
        self.historical_weather_scheduler = FakeScheduler()
        self.forecast_scheduler = FakeScheduler()

    def close(self) -> None:
        self.home_assistant.close()


def build_test_app(
    *,
    imported_rows: dict[str, int] | None = None,
    sensors: dict[str, str] | None = None,
) -> tuple[FastAPI, FakeHomeAssistantGateway]:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows=imported_rows or {}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_max_days_back": 14,
            "sensors": sensors or {"room_temperature": "sensor.room_temperature"},
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )
    return app, gateway


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(20):
        response = client.get(f"/api/history-import/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        sleep(0.05)

    raise AssertionError("history import job did not finish")


def test_dashboard_shows_import_button_without_simulation_link() -> None:
    app, gateway = build_test_app(imported_rows={"room_temperature": 3})

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Importeer geschiedenis" in response.text
    assert "Importeer weerdata" in response.text
    assert 'static/shared.css' not in response.text
    assert 'static/shared.js' not in response.text
    assert 'static/dashboard.css' in response.text
    assert 'static/dashboard.js' in response.text
    assert 'href="simulation"' not in response.text
    assert app.state.container.telemetry_scheduler.started is True
    assert app.state.container.historical_weather_scheduler.started is True
    assert app.state.container.forecast_scheduler.started is True
    assert gateway.closed is True


def test_removed_routes_return_404() -> None:
    app, _ = build_test_app()

    with TestClient(app) as client:
        assert client.get("/simulation").status_code == 404
        assert client.get("/api/identification").status_code == 404
        assert client.post("/api/prediction", json={}).status_code == 404
        assert client.post("/api/mpc/thermostat-setpoint", json={}).status_code == 404


def test_weather_import_endpoint_runs_forecast_backfill() -> None:
    app, _ = build_test_app(imported_rows={"room_temperature": 3})

    with TestClient(app) as client:
        response = client.post("/api/weather-import")

    assert response.status_code == 200
    assert response.json() == {"imported_rows": 12}
    assert app.state.container.weather_import_service.import_calls == 1


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
    app, gateway = build_test_app(
        imported_rows={"room_temperature": 3, "outdoor_temperature": 7},
        sensors={
            "room_temperature": "sensor.room_temperature",
            "outdoor_temperature": "sensor.outdoor_temperature",
        },
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")
        job = wait_for_job(client, response.json()["job_id"])

    assert response.status_code == 200
    assert response.json()["sensor_count"] == 2
    assert job["status"] == "succeeded"
    assert gateway.closed is True


def test_history_import_job_endpoint_returns_result() -> None:
    app, _ = build_test_app(
        imported_rows={"room_temperature": 3, "outdoor_temperature": 7},
        sensors={
            "room_temperature": "sensor.room_temperature",
            "outdoor_temperature": "sensor.outdoor_temperature",
        },
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")
        assert response.status_code == 200
        payload = wait_for_job(client, response.json()["job_id"])

    assert payload["status"] == "succeeded"
    assert payload["imported_rows"] == {"room_temperature": 3, "outdoor_temperature": 7}
    assert payload["total_rows"] == 10
    assert payload["sensor_count"] == 2
    assert payload["error"] is None


def test_dashboard_charts_endpoint_returns_day_series() -> None:
    chart_date = date(2026, 4, 25)
    app, _ = build_test_app(imported_rows={})

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
    assert payload["forecast_temperature"] == {
        "name": "temperature",
        "unit": "degC",
        "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 12.5}],
    }
    assert payload["historical_weather_gti"][1]["points"][:3] == [
        {"timestamp": "2026-04-25T12:00:00+00:00", "value": 210.0},
        {"timestamp": "2026-04-25T12:15:00+00:00", "value": 210.0},
        {"timestamp": "2026-04-25T12:30:00+00:00", "value": 210.0},
    ]
    local_timezone = dashboard_charts_module.current_timezone()
    start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
    end_time = start_time + timedelta(days=1)
    forecast_end_time = end_time + timedelta(minutes=15)
    assert app.state.container.time_series_read_repository.calls == [
        ("numeric", ["shutter_living_room"], start_time.isoformat(), end_time.isoformat()),
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
        ("text", ["hp_mode"], start_time.isoformat(), end_time.isoformat()),
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
    app, _ = build_test_app(imported_rows={})
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


def test_plotly_script_is_served_locally() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        response = client.get("/plotly.js")

    assert response.status_code == 200
    assert "Plotly" in response.text[:5000]


def test_javascript_and_css_are_not_cached() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        responses = [
            client.get("/static/dashboard.js"),
            client.get("/static/dashboard.css"),
            client.get("/plotly.js"),
        ]

    for response in responses:
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"
