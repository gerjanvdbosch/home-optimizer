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
from home_optimizer.domain.time import current_local_timezone
from home_optimizer.features import HistoryImportResult
from home_optimizer.features.modeling import StoredRoomModelVersion
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


class FakeModelVersionRepository:
    def __init__(self) -> None:
        self.saved_versions: list[StoredRoomModelVersion] = []

    def save_room_model_version(self, version: StoredRoomModelVersion) -> None:
        self.saved_versions.append(version)


def build_half_hourly_series(
    *,
    name: str,
    unit: str,
    start_value: float,
    changed_value: float | None = None,
    change_index: int | None = None,
    start_time: datetime | None = None,
) -> NumericSeries:
    start_time = start_time or datetime(
        2026,
        4,
        25,
        0,
        0,
        tzinfo=current_local_timezone(),
    ).astimezone(ZoneInfo("UTC"))
    points: list[NumericPoint] = []
    for index in range(48):
        value = start_value
        if changed_value is not None and change_index is not None and index >= change_index:
            value = changed_value
        points.append(
            NumericPoint(
                timestamp=(
                    start_time + timedelta(minutes=30 * index)
                ).isoformat(timespec="seconds"),
                value=value,
            )
        )
    return NumericSeries(name=name, unit=unit, points=points)


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
            build_half_hourly_series(
                name="room_temperature",
                unit="°C",
                start_value=20.5,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
            NumericSeries(
                name="outdoor_temperature",
                unit="°C",
                points=build_half_hourly_series(
                    name="outdoor_temperature",
                    unit="°C",
                    start_value=12.1,
                    start_time=start_time.astimezone(ZoneInfo("UTC")),
                ).points,
            ),
            build_half_hourly_series(
                name="thermostat_setpoint",
                unit="°C",
                start_value=20.0,
                changed_value=21.0,
                change_index=24,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
            build_half_hourly_series(
                name="dhw_top_temperature",
                unit="°C",
                start_value=48.0,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
            NumericSeries(
                name="dhw_bottom_temperature",
                unit="°C",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=42.0)],
            ),
            build_half_hourly_series(
                name="hp_electric_power",
                unit="kW",
                start_value=1.5,
                changed_value=0.0,
                change_index=24,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
            build_half_hourly_series(
                name="compressor_frequency",
                unit="Hz",
                start_value=0.0,
                changed_value=40.0,
                change_index=12,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
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
            build_half_hourly_series(
                name="p1_net_power",
                unit="kW",
                start_value=2.0,
                changed_value=-0.5,
                change_index=24,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
            build_half_hourly_series(
                name="pv_output_power",
                unit="kW",
                start_value=0.0,
                changed_value=1.0,
                change_index=24,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
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
                unit="°C",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=12.5)],
            ),
            NumericSeries(
                name="gti_pv",
                unit="W/m2",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=500.0)],
            ),
            build_half_hourly_series(
                name="gti_living_room_windows",
                unit="W/m2",
                start_value=220.0,
                start_time=start_time.astimezone(ZoneInfo("UTC")),
            ),
        ]

    def read_electricity_price_series(        self,
        start_time,
        end_time,
        *,
        source,
        interval_minutes=15,
    ) -> NumericSeries:
        self.calls.append(
            ("electricity_price", [source], start_time.isoformat(), end_time.isoformat())
        )
        return NumericSeries(
            name="electricity_price",
            unit="EUR/kWh",
            points=[
                NumericPoint(
                    timestamp=start_time.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds"),
                    value=0.245,
                ),
                NumericPoint(
                    timestamp=(
                        start_time.astimezone(ZoneInfo("UTC")) + timedelta(hours=12)
                    ).isoformat(timespec="seconds"),
                    value=0.245,
                ),
            ],
        )


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
        self.model_version_repository = FakeModelVersionRepository()
        self.telemetry_scheduler = FakeScheduler()
        self.electricity_price_scheduler = FakeScheduler()
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
            "room_target": [
                {"time": "00:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
                {"time": "08:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
                {"time": "14:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
                {"time": "18:00", "target": 20.0, "low_margin": 0.5, "high_margin": 1.5},
                {"time": "22:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            ],
            "dhw_target": [
                {"time": "00:00", "target": 20.0, "low_margin": 5.0, "high_margin": 30.0},
                {"time": "10:00", "target": 20.0, "low_margin": 5.0, "high_margin": 35.0},
                {"time": "19:59", "target": 20.0, "low_margin": 5.0, "high_margin": 35.0},
                {"time": "20:00", "target": 50.0, "low_margin": 2.0, "high_margin": 5.0},
                {"time": "21:00", "target": 50.0, "low_margin": 2.0, "high_margin": 5.0},
                {"time": "21:01", "target": 20.0, "low_margin": 5.0, "high_margin": 30.0},
            ],
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
    assert app.state.container.electricity_price_scheduler.started is True
    assert app.state.container.forecast_scheduler.started is True
    assert gateway.closed is True


def test_removed_routes_return_404() -> None:
    app, _ = build_test_app()

    with TestClient(app) as client:
        assert client.get("/simulation").status_code == 404
        assert client.post("/api/prediction", json={}).status_code == 404
        assert client.post("/api/mpc/thermostat-setpoint", json={}).status_code == 404


def test_identification_endpoint_returns_dataset_and_summary() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        response = client.get(
            "/api/identification",
            params={
                "start_time": "2026-04-25T00:00:00+00:00",
                "end_time": "2026-04-25T01:00:00+00:00",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["interval_minutes"] == 15
    assert payload["summary"]["total_rows"] == 4
    assert payload["summary"]["mode_space_rows"] == 0
    assert payload["summary"]["mode_off_rows"] == 4
    assert payload["summary"]["defrost_rows"] == 0
    assert payload["summary"]["booster_rows"] == 0
    assert payload["summary"]["valid_room_rows"] == 4
    assert payload["summary"]["valid_dhw_rows"] == 4
    assert payload["summary"]["valid_cop_rows"] == 0
    assert (
        payload["summary"]["exclusion_reason_counts"]["missing_or_nonpositive_thermal_output"]
        == 4
    )
    assert len(payload["rows"]) == 4
    assert payload["rows"][0]["mode_space"] == 0
    assert payload["rows"][0]["mode_dhw"] == 0
    assert payload["rows"][0]["mode_off"] == 1
    assert payload["rows"][0]["booster_heater_active"] == 0
    assert payload["rows"][0]["dhw_draw_proxy_c"] == 0.0
    assert payload["rows"][0]["is_valid_for_room_identification"] is True


def test_weather_import_endpoint_runs_forecast_backfill() -> None:
    app, _ = build_test_app(imported_rows={"room_temperature": 3})

    with TestClient(app) as client:
        response = client.post("/api/weather-import")

    assert response.status_code == 200
    assert response.json() == {"imported_rows": 12}
    assert app.state.container.weather_import_service.import_calls == 1


def test_train_endpoint_trains_and_stores_room_model_version() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        response = client.post(
            "/api/train",
            params={
                "start_time": "2026-04-25T00:00:00+00:00",
                "end_time": "2026-04-26T00:00:00+00:00",
                "interval_minutes": 15,
                "min_train_rows": 20,
                "validation_window_rows": 24,
                "activate": "true",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"].startswith("room-model-")
    assert payload["model_type"] == "linear_room"
    assert payload["interval_minutes"] == 15
    assert payload["sample_count"] > 0
    assert payload["is_active"] is True
    assert len(payload["aggregate_metrics"]) == 5
    assert app.state.container.model_version_repository.saved_versions
    assert app.state.container.model_version_repository.saved_versions[0].is_active is True

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
    local_timezone = dashboard_charts_module.current_timezone()
    expected_day_start = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
    expected_day_end = expected_day_start + timedelta(days=1)
    assert payload["electricity_price"] == {
        "name": "electricity_price",
        "unit": "EUR/kWh",
        "points": [
            {
                "timestamp": expected_day_start.astimezone(ZoneInfo("UTC")).isoformat(
                    timespec="seconds"
                ),
                "value": 0.245,
            },
            {
                "timestamp": (expected_day_start + timedelta(hours=12))
                .astimezone(ZoneInfo("UTC"))
                .isoformat(timespec="seconds"),
                "value": 0.245,
            },
        ],
    }
    assert payload["room_temperature"]["points"][0] == {
        "timestamp": expected_day_start.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "value": 20.5,
    }
    assert payload["room_temperature"]["points"][24] == {
        "timestamp": (expected_day_start + timedelta(hours=12))
        .astimezone(ZoneInfo("UTC"))
        .isoformat(timespec="seconds"),
        "value": 20.5,
    }
    assert payload["outdoor_temperature"] == {
        "name": "outdoor_temperature",
        "unit": "°C",
        "points": payload["outdoor_temperature"]["points"],
    }
    assert payload["outdoor_temperature"]["points"][24]["value"] == 12.1
    assert payload["thermostat_setpoint"]["name"] == "thermostat_setpoint"
    assert payload["thermostat_setpoint"]["unit"] == "°C"
    assert payload["thermostat_setpoint"]["points"][0] == {
        "timestamp": expected_day_start.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "value": 20.0,
    }
    assert payload["thermostat_setpoint"]["points"][24] == {
        "timestamp": (expected_day_start + timedelta(hours=12))
        .astimezone(ZoneInfo("UTC"))
        .isoformat(timespec="seconds"),
        "value": 21.0,
    }
    assert payload["room_target_temperature"]["points"][:2] == [
        {
            "timestamp": expected_day_start.astimezone(ZoneInfo("UTC")).isoformat(
                timespec="seconds"
            ),
            "value": 19.0,
        },
        {
            "timestamp": (expected_day_start + timedelta(minutes=15))
            .astimezone(ZoneInfo("UTC"))
            .isoformat(timespec="seconds"),
            "value": 19.0,
        },
    ]
    assert payload["room_target_min_temperature"]["points"][0]["value"] == 18.5
    assert payload["room_target_max_temperature"]["points"][-1] == {
        "timestamp": (expected_day_end - timedelta(minutes=15))
        .astimezone(ZoneInfo("UTC"))
        .isoformat(timespec="seconds"),
        "value": 20.5,
    }
    assert [series["name"] for series in payload["dhw_temperatures"]] == [
        "dhw_top_temperature",
        "dhw_bottom_temperature",
    ]
    assert payload["dhw_target_temperature"]["points"][80] == {
        "timestamp": datetime.combine(chart_date, time(20, 0), tzinfo=local_timezone)
        .astimezone(ZoneInfo("UTC"))
        .isoformat(timespec="seconds"),
        "value": 50.0,
    }
    assert payload["dhw_target_min_temperature"]["points"][80]["value"] == 48.0
    assert payload["dhw_target_max_temperature"]["points"][80]["value"] == 55.0
    assert payload["forecast_temperature"] == {
        "name": "temperature",
        "unit": "°C",
        "points": [{"timestamp": "2026-04-25T12:00:00+00:00", "value": 12.5}],
    }
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
                "outdoor_temperature",
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
            "electricity_price",
            ["nordpool"],
            start_time.isoformat(),
            end_time.isoformat(),
        ),
    ]


def test_dashboard_kpis_endpoint_returns_daily_metrics() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        response = client.get("/api/kpis?date=2026-04-25")

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_valid_for_control_evaluation"] is True
    assert payload["validity_reasons"] == []
    assert payload["data_coverage_pct"] == 100.0
    assert payload["largest_data_gap_minutes"] == 30.0
    assert payload["hp_electric_kwh"] is not None
    assert payload["total_import_kwh"] is not None
    assert payload["total_export_kwh"] is not None
    assert payload["pv_generation_kwh"] is not None
    assert payload["solar_irradiance_mean_w_m2"] == 220.0
    assert payload["shutter_open_pct_mean"] == 50.0
    assert payload["outdoor_temperature_mean_c"] == pytest.approx(12.1)
    assert payload["self_consumption_ratio"] is not None
    assert payload["electricity_cost_eur"] is not None
    assert payload["room_comfort_undershoot_degree_hours"] is not None
    assert payload["comfort_overshoot_while_heating_degree_hours"] is not None
    assert payload["comfort_overshoot_passive_degree_hours"] is not None
    assert payload["dhw_comfort_undershoot_minutes"] is not None
    assert payload["thermostat_setpoint_changes"] == 1
    assert payload["compressor_starts"] == 1


def test_baseline_kpi_summary_endpoint_uses_default_date_range() -> None:
    app, _ = build_test_app(imported_rows={})

    with TestClient(app) as client:
        response = client.get("/api/kpi-summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["number_of_days"] == 20
    assert payload["number_of_valid_days"] == 20
    assert payload["mean_hp_electric_kwh_per_day"] is not None
    assert payload["mean_electricity_cost_eur_per_day"] is not None
    assert payload["mean_room_temperature_mae_c"] is not None
    assert payload["mean_solar_irradiance_w_m2"] == 220.0
    assert payload["mean_shutter_open_pct"] == 50.0
    assert payload["total_comfort_undershoot_degree_hours"] >= 0.0
    assert payload["total_comfort_overshoot_while_heating_degree_hours"] >= 0.0
    assert payload["total_comfort_overshoot_passive_degree_hours"] >= 0.0
    assert payload["total_dhw_undershoot_minutes"] >= 0.0
    assert payload["mean_compressor_starts_per_day"] is not None
    assert payload["mean_self_consumption_ratio"] is not None


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
