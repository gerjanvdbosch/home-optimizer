"""API tests for the unified MPC web interface."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from home_optimizer.api import HomeOptimizerAPI, app
from home_optimizer.optimizer import Optimizer, RunRequest
from home_optimizer.telemetry import TelemetryRepository
from home_optimizer.types import CalibrationParameterOverrides, CalibrationSnapshotPayload

client = TestClient(app)


def test_simulate_exposes_pv_forecast_in_api_response() -> None:
    """The simulate endpoint must return explicit PV forecast data and chart JSON."""
    response = client.post(
        "/api/simulate",
        json={
            "pv_enabled": True,
            "pv_peak_kw": 4.0,
            "horizon_hours": 24,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["pv_enabled"] is True
    assert payload["first_ufh_power_kw"] >= 0.0
    assert payload["first_total_hp_power_kw"] >= payload["first_ufh_power_kw"]
    assert len(payload["control_labels"]) == 24
    assert len(payload["pv_forecast_kw"]) == 24
    assert max(payload["pv_forecast_kw"]) > 0.0
    assert payload["pv_forecast_fig"]

    pv_fig = json.loads(payload["pv_forecast_fig"])
    assert pv_fig["data"][0]["name"] == "P<sub>PV</sub> forecast [kW]"


def test_simulate_supports_combined_mode_through_unified_mpc() -> None:
    """The same simulate endpoint must solve UFH + DHW via the unified MPC controller."""
    response = client.post(
        "/api/simulate",
        json={
            "dhw_enabled": True,
            "pv_enabled": True,
            "pv_peak_kw": 3.0,
            "horizon_hours": 8,
            "dhw_v_tap_m3_per_h": 0.015,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["dhw_enabled"] is True
    assert payload["first_dhw_power_kw"] >= 0.0
    assert payload["dhw_fig"]
    assert payload["power_fig"]
    assert len(payload["control_labels"]) == 8
    assert len(payload["pv_forecast_kw"]) == 8
    assert payload["max_dhw_comfort_violation_c"] >= 0.0


def test_dashboard_html_contains_dhw_and_pv_sections() -> None:
    """The simulator page must expose grouped DHW/PV UI elements for the unified controller."""
    response = client.get("/simulator")

    assert response.status_code == 200
    html = response.text
    assert 'id="dhw_enabled"' in html
    assert 'id="dhw-settings"' in html
    assert 'id="dhw-chart-card"' in html
    assert 'id="pv_enabled"' in html
    assert "UFH + DHW + PV MPC" in html


def test_dashboard_html_contains_optimizer_mpc_forecast_section() -> None:
    """Dashboard page must expose Optimizer MPC forecast KPIs and chart containers."""
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "MPC voorspelling (Optimizer)" in html
    assert 'id="kpi-mpc-status"' in html
    assert 'id="kpi-mpc-ufh-p0"' in html
    assert 'id="kpi-mpc-dhw-p0"' in html
    assert 'id="kpi-mpc-cost"' in html
    assert 'id="chart-mpc-temp"' in html
    assert 'id="chart-mpc-power"' in html
    assert "/api/optimizer/latest" in html


def test_dashboard_html_uses_api_solar_figure_for_gti_chart() -> None:
    """Dashboard JS must render the GTI chart from the forecast API figure JSON."""
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "JSON.parse(d.solar_forecast_fig)" in html
    assert "Plotly.react('chart-solar', solarFig.data, solarFig.layout, CFG);" in html
    assert "function buildSolarFig(labels, gtiWindow, gtiPv)" not in html


def test_optimizer_latest_returns_404_without_scheduled_run() -> None:
    """Operational endpoint must fail clearly until a periodic run has succeeded."""
    Optimizer.clear_latest_scheduled_snapshot()
    response = client.get("/api/optimizer/latest")
    assert response.status_code == 404


def test_optimizer_latest_returns_scheduled_optimizer_result() -> None:
    """Operational endpoint must expose the latest cached periodic Optimizer run."""
    Optimizer.clear_latest_scheduled_snapshot()
    optimizer = Optimizer()
    req = RunRequest.model_validate({"horizon_hours": 8})
    result = optimizer.run_scheduled_once(base_input=req)
    assert result is not None

    response = client.get("/api/optimizer/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]
    assert payload["first_ufh_power_kw"] >= 0.0
    assert len(payload["control_labels"]) == 8


def test_optimizer_scheduled_input_applies_latest_calibration_snapshot(tmp_path) -> None:
    """Scheduled MPC input must apply the latest persisted calibration overrides before solving."""
    database_url = f"sqlite:///{tmp_path / 'optimizer-calibration.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                C_r=7.2,
                C_b=11.3,
                eta_carnot=0.39,
                T_supply_min=26.5,
            ),
        )
    )

    base_input = RunRequest.model_validate({"horizon_hours": 8, "C_r": 6.0, "eta_carnot": 0.45})
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=None,
        repository=repository,
    )

    assert scheduled_input.C_r == 7.2
    assert scheduled_input.C_b == 11.3
    assert scheduled_input.eta_carnot == 0.39
    assert scheduled_input.T_supply_min == 26.5


def test_calibration_latest_returns_404_without_snapshot(monkeypatch, tmp_path) -> None:
    """Calibration endpoint must fail clearly until the first automatic snapshot exists."""
    database_url = f"sqlite:///{tmp_path / 'calibration-api-empty.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/calibration/latest")

    assert response.status_code == 404


def test_calibration_latest_returns_latest_snapshot(monkeypatch, tmp_path) -> None:
    """Calibration endpoint must expose the latest persisted automatic calibration snapshot."""
    database_url = f"sqlite:///{tmp_path / 'calibration-api.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                R_ro=9.1,
                eta_carnot=0.42,
                dhw_R_loss=55.0,
            ),
        )
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/calibration/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["effective_parameters"]["R_ro"] == 9.1
    assert payload["effective_parameters"]["eta_carnot"] == 0.42
    assert payload["effective_parameters"]["dhw_R_loss"] == 55.0


def test_latest_forecast_api_keeps_pv_trace_visible_even_for_zero_pv_gti(
    tmp_path,
    monkeypatch,
) -> None:
    """Latest-forecast solar figure must expose both GTI series, including an all-zero PV profile."""
    database_url = f"sqlite:///{tmp_path / 'forecast-api.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    fetched_at = datetime(2026, 4, 17, 19, 0, tzinfo=timezone.utc)
    repository.bulk_add_forecast_snapshots(
        [
            {
                "fetched_at_utc": fetched_at,
                "valid_at_utc": fetched_at + timedelta(hours=step_k),
                "step_k": step_k,
                "dt_hours": 1.0,
                "t_out_c": 8.0 - step_k,
                "gti_w_per_m2": value,
                "gti_pv_w_per_m2": 0.0,
            }
            for step_k, value in enumerate([120.0, 60.0, 0.0])
        ]
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/forecast/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["gti_pv_w_per_m2"] == [0.0, 0.0, 0.0]

    solar_fig = json.loads(payload["solar_forecast_fig"])
    assert [trace["name"] for trace in solar_fig["data"]] == [
        "GTI ramen [W/m2]",
        "GTI PV-panelen [W/m2]",
    ]
    assert solar_fig["data"][1]["y"] == [0.0, 0.0, 0.0]


