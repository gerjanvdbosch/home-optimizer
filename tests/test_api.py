"""API tests for the unified MPC web interface."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from home_optimizer.api import app

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
    """The dashboard page must expose grouped DHW/PV UI elements for the unified controller."""
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="dhw_enabled"' in html
    assert 'id="dhw-settings"' in html
    assert 'id="dhw-chart-card"' in html
    assert 'id="pv_enabled"' in html
    assert 'UFH + DHW + PV MPC' in html

