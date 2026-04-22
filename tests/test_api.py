"""API tests for the unified MPC web interface."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import home_optimizer.application.optimizer as optimizer_module

from home_optimizer.api import HomeOptimizerAPI, api_service, app
from home_optimizer.application.optimizer import Optimizer, RunRequest
from home_optimizer.sensors import LiveReadings, SensorBackend
from home_optimizer.telemetry import TelemetryRepository
from home_optimizer.types import CalibrationParameterOverrides, CalibrationSnapshotPayload, CalibrationStageResult

client = TestClient(app)


def _live_readings(*, shutter_living_room_pct: float) -> LiveReadings:
    """Return one complete live sensor snapshot for scheduled-input tests."""
    return LiveReadings(
        room_temperature_c=20.5,
        outdoor_temperature_c=8.0,
        hp_supply_temperature_c=31.0,
        hp_supply_target_temperature_c=33.0,
        hp_return_temperature_c=27.0,
        hp_flow_lpm=9.0,
        hp_electric_power_kw=2.0,
        hp_mode="ufh",
        p1_net_power_kw=1.4,
        pv_output_kw=0.6,
        thermostat_setpoint_c=20.5,
        dhw_top_temperature_c=52.0,
        dhw_bottom_temperature_c=45.0,
        shutter_living_room_pct=shutter_living_room_pct,
        defrost_active=False,
        booster_heater_active=False,
        boiler_ambient_temp_c=18.0,
        refrigerant_condensation_temp_c=38.0,
        refrigerant_liquid_line_temp_c=28.0,
        discharge_temp_c=65.0,
        t_mains_estimated_c=10.5,
        timestamp=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc),
        pv_total_kwh=1000.0,
        hp_electric_total_kwh=500.0,
        p1_import_total_kwh=800.0,
        p1_export_total_kwh=200.0,
    )


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
            "dhw_v_tap_forecast": [0.015] * 8,
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


def test_simulate_passes_safe_calibration_overrides_into_ml_forecast_generation(
    monkeypatch,
    tmp_path,
) -> None:
    """API simulation must materialise calibrated DHW parameters before ML forecast providers run.

    DHW tap-profile artifacts are keyed by the effective physical tank tuple. When
    the simulate endpoint skipped calibration overrides, the ML forecaster saw the
    static request defaults instead of the calibrated runtime tuple and therefore
    refused to reuse the persisted DHW artifact. This regression asserts that the
    calibrated fields are now present in both ``current_overrides`` and the fully
    materialised request passed into the forecast service.
    """

    database_url = f"sqlite:///{tmp_path / 'simulate-calibrated-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    fetched_at = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    repository.bulk_add_forecast_snapshots(
        [
            {
                "fetched_at_utc": fetched_at,
                "valid_at_utc": fetched_at + timedelta(hours=step_k),
                "step_k": step_k,
                "dt_hours": 1.0,
                "t_out_c": 8.0,
                "gti_w_per_m2": 0.0,
                "gti_pv_w_per_m2": 0.0,
            }
            for step_k in range(4)
        ]
    )
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                dhw_R_loss=92.0,
                dhw_boiler_ambient_bias_c=4.0,
            ),
        )
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    captured: dict[str, dict[str, object]] = {}

    def _fake_build_missing_overrides(*, request_data, repository, weather_rows, current_overrides=None):  # noqa: ANN001
        captured["request_data"] = dict(request_data)
        captured["current_overrides"] = dict(current_overrides or {})
        return {"dhw_v_tap_forecast": [0.0, 0.0, 0.0, 0.0]}

    monkeypatch.setattr(optimizer_module._FORECAST_SERVICE, "build_missing_overrides", _fake_build_missing_overrides)

    response = client.post(
        "/api/simulate",
        json={
            "horizon_hours": 4,
            "dhw_enabled": True,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
        },
    )

    assert response.status_code == 200
    assert captured["current_overrides"]["dhw_R_loss"] == 92.0
    assert captured["current_overrides"]["dhw_boiler_ambient_bias_c"] == 4.0
    assert captured["request_data"]["dhw_R_loss"] == 92.0
    assert captured["request_data"]["dhw_boiler_ambient_bias_c"] == 4.0


def test_dashboard_html_contains_dhw_and_pv_sections() -> None:
    """The simulator page must expose grouped DHW/PV UI elements for the unified controller."""
    response = client.get("/simulator")

    assert response.status_code == 200
    html = response.text
    assert 'id="dhw_enabled"' in html
    assert 'id="dhw-settings"' in html
    assert 'id="dhw-chart-card"' in html
    assert 'id="pv_enabled"' in html
    assert 'id="shutter_living_room_pct"' in html
    assert "UFH + DHW + PV MPC" in html
    assert "fetch(apiUrl('/api/defaults'))" in html
    assert "applyRunRequestDefaults" in html


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


def test_run_request_exposes_canonical_internal_gains_heat_fraction_field() -> None:
    """RunRequest must only expose the canonical internal_gains_heat_fraction field name."""
    request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "internal_gains_heat_fraction": 0.45,
        }
    )

    assert request.internal_gains_heat_fraction == 0.45
    dumped = request.model_dump(mode="python")
    assert dumped["internal_gains_heat_fraction"] == 0.45
    assert "baseload_internal_gains_heat_fraction" not in dumped


def test_optimizer_latest_returns_404_without_scheduled_run() -> None:
    """Operational endpoint must fail clearly until a periodic run has succeeded."""
    Optimizer.clear_latest_scheduled_snapshot()
    response = client.get("/api/optimizer/latest")
    assert response.status_code == 404


def test_optimizer_latest_returns_scheduled_optimizer_result() -> None:
    """Operational endpoint must expose the latest cached periodic Optimizer run."""
    Optimizer.clear_latest_scheduled_snapshot()
    optimizer = Optimizer()
    req = RunRequest.model_validate({"horizon_hours": 8, "dhw_v_tap_forecast": [0.0] * 8})
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


def test_optimizer_scheduled_input_ignores_invalid_ufh_calibration_tuple(tmp_path) -> None:
    """Scheduled MPC must ignore unsafe calibrated UFH tuples but keep safe groups."""
    database_url = f"sqlite:///{tmp_path / 'optimizer-unsafe-calibration.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                C_r=6.0,
                C_b=0.15625,
                R_br=20.8625,
                R_ro=1.7483,
                dhw_R_loss=55.0,
            ),
        )
    )

    base_input = RunRequest.model_validate({"horizon_hours": 8, "dt_hours": 1.0})
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=None,
        repository=repository,
    )

    assert scheduled_input.C_r == base_input.C_r
    assert scheduled_input.C_b == base_input.C_b
    assert scheduled_input.R_br == base_input.R_br
    assert scheduled_input.R_ro == base_input.R_ro
    assert scheduled_input.dhw_R_loss == 55.0


def test_optimizer_scheduled_input_applies_live_shutter_position() -> None:
    """Scheduled MPC input must use the latest shutter reading to scale UFH solar gains."""

    class StaticBackend(SensorBackend):
        def __init__(self, readings: LiveReadings) -> None:
            self._readings = readings

        def read_all(self) -> LiveReadings:
            return self._readings

        def close(self) -> None:
            return None

    base_input = RunRequest.model_validate({"horizon_hours": 8, "shutter_living_room_pct": 100.0})
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=StaticBackend(_live_readings(shutter_living_room_pct=35.0)),
        repository=None,
    )

    assert scheduled_input.shutter_living_room_pct == 35.0


def test_optimizer_scheduled_input_applies_persisted_sensor_biases_to_live_readings(tmp_path) -> None:
    """Scheduled MPC input must correct live sensor readings with persisted bias overrides."""

    class StaticBackend(SensorBackend):
        def __init__(self, readings: LiveReadings) -> None:
            self._readings = readings

        def read_all(self) -> LiveReadings:
            return self._readings

        def close(self) -> None:
            return None

    database_url = f"sqlite:///{tmp_path / 'optimizer-sensor-bias.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                room_temperature_bias_c=0.7,
                dhw_top_temperature_bias_c=1.2,
                dhw_bottom_temperature_bias_c=-0.8,
                dhw_boiler_ambient_bias_c=0.5,
            ),
        )
    )

    raw_readings = _live_readings(shutter_living_room_pct=35.0)
    base_input = RunRequest.model_validate({"horizon_hours": 8})
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=StaticBackend(raw_readings),
        repository=repository,
    )

    assert scheduled_input.T_r_init == raw_readings.room_temperature_c + 0.7
    assert scheduled_input.dhw_T_top_init == raw_readings.dhw_top_temperature_c + 1.2
    assert scheduled_input.dhw_T_bot_init == raw_readings.dhw_bottom_temperature_c - 0.8
    assert scheduled_input.dhw_t_amb_c == raw_readings.boiler_ambient_temp_c + 0.5


def test_optimizer_scheduled_input_keeps_explicit_shutter_forecast(tmp_path) -> None:
    """A caller-supplied shutter forecast must survive scheduled live-sensor overrides."""

    class StaticBackend(SensorBackend):
        def __init__(self, readings: LiveReadings) -> None:
            self._readings = readings

        def read_all(self) -> LiveReadings:
            return self._readings

        def close(self) -> None:
            return None

    database_url = f"sqlite:///{tmp_path / 'optimizer-shutter-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    base_input = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "shutter_living_room_pct": 100.0,
            "shutter_forecast": [100.0, 80.0, 40.0, 0.0],
        }
    )
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=StaticBackend(_live_readings(shutter_living_room_pct=35.0)),
        repository=repository,
    )

    assert scheduled_input.shutter_living_room_pct == 35.0
    assert scheduled_input.shutter_forecast == [100.0, 80.0, 40.0, 0.0]


def test_simulate_accepts_explicit_shutter_forecast() -> None:
    """The API must accept a real horizon-wide shutter forecast supplied by the caller."""
    response = client.post(
        "/api/simulate",
        json={
            "horizon_hours": 4,
            "dt_hours": 1.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [600.0, 600.0, 600.0, 600.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "shutter_living_room_pct": 100.0,
            "shutter_forecast": [100.0, 50.0, 25.0, 0.0],
            "pv_enabled": False,
            "dhw_enabled": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]
    assert len(payload["control_labels"]) == 4


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
            ufh_active=CalibrationStageResult(
                stage_name="ufh_active",
                succeeded=False,
                message="Automatic UFH RC fit rejected: optimiser converged to parameter bounds.",
                diagnostics={
                    "selected_segment_count": 1,
                    "required_min_selected_segments": 2,
                    "bound_violations": ["C_b at lower bound"],
                },
            ),
            dhw_active=CalibrationStageResult(
                stage_name="dhw_active",
                succeeded=False,
                message="Automatic DHW active fit rejected: insufficient active DHW excitation.",
                diagnostics={
                    "selected_segment_count": 1,
                    "required_min_selected_segments": 2,
                    "fitted_r_strat_k_per_kw": 12.0,
                },
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
    assert payload["ufh_active"]["diagnostics"]["bound_violations"] == ["C_b at lower bound"]
    assert payload["dhw_active"]["diagnostics"]["required_min_selected_segments"] == 2


def test_defaults_returns_static_runrequest_without_calibration_snapshot(monkeypatch, tmp_path) -> None:
    """/api/defaults must fall back to plain RunRequest defaults when no snapshot exists."""
    database_url = f"sqlite:///{tmp_path / 'defaults-empty.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/defaults")

    assert response.status_code == 200
    payload = response.json()
    defaults = RunRequest.model_validate({})
    assert payload["C_r"] == defaults.C_r
    assert payload["eta_carnot"] == defaults.eta_carnot
    assert payload["T_supply_min"] == defaults.T_supply_min


def test_defaults_returns_latest_calibration_snapshot_over_static_defaults(monkeypatch, tmp_path) -> None:
    """/api/defaults must expose calibrated values when a snapshot is available."""
    database_url = f"sqlite:///{tmp_path / 'defaults-calibrated.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                C_r=7.4,
                C_b=11.2,
                eta_carnot=0.38,
                T_supply_min=26.8,
            ),
        )
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/defaults")

    assert response.status_code == 200
    payload = response.json()
    assert payload["C_r"] == 7.4
    assert payload["C_b"] == 11.2
    assert payload["eta_carnot"] == 0.38
    assert payload["T_supply_min"] == 26.8


def test_defaults_prefers_registered_runtime_base_request(monkeypatch, tmp_path) -> None:
    """`/api/defaults` must expose the configured runtime base request, not bare model defaults."""

    database_url = f"sqlite:///{tmp_path / 'defaults-runtime-base.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    original_base_request = api_service._base_request
    try:
        runtime_base_request = RunRequest.model_validate(
            {
                "horizon_hours": 8,
                "dhw_C_top": 0.11628,
                "dhw_C_bot": 0.11628,
                "dhw_R_loss": 80.0,
            }
        )
        api_service.set_base_request(runtime_base_request)

        response = client.get("/api/defaults")

        assert response.status_code == 200
        payload = response.json()
        assert payload["dhw_C_top"] == runtime_base_request.dhw_C_top
        assert payload["dhw_C_bot"] == runtime_base_request.dhw_C_bot
        assert payload["dhw_R_loss"] == runtime_base_request.dhw_R_loss
    finally:
        api_service.set_base_request(original_base_request)


def test_defaults_ignores_invalid_ufh_calibration_tuple_but_keeps_safe_groups(monkeypatch, tmp_path) -> None:
    """/api/defaults must not expose calibrated values that fail runtime validation."""
    database_url = f"sqlite:///{tmp_path / 'defaults-unsafe-calibrated.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    repository.add_calibration_snapshot(
        CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                C_r=6.0,
                C_b=0.15625,
                R_br=20.8625,
                R_ro=1.7483,
                dhw_R_loss=55.0,
            ),
        )
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    response = client.get("/api/defaults")

    assert response.status_code == 200
    payload = response.json()
    defaults = RunRequest.model_validate({})
    assert payload["C_r"] == defaults.C_r
    assert payload["C_b"] == defaults.C_b
    assert payload["R_br"] == defaults.R_br
    assert payload["R_ro"] == defaults.R_ro
    assert payload["dhw_R_loss"] == 55.0


def test_simulate_uses_registered_runtime_base_request_for_dhw_forecast_lookup(monkeypatch, tmp_path) -> None:
    """`/api/simulate` must materialize the configured DHW tank before ML forecast lookup.

    The persisted DHW tap-profile artifact is keyed by the effective DHW physics.
    When the API started from plain `RunRequest` defaults, the artifact lookup
    missed because `dhw_C_top`/`dhw_C_bot` did not match the real configured tank.
    This regression verifies that the registered runtime base request is now what
    the forecast provider sees.
    """

    database_url = f"sqlite:///{tmp_path / 'simulate-runtime-dhw-base.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
    fetched_at = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    repository.bulk_add_forecast_snapshots(
        [
            {
                "fetched_at_utc": fetched_at,
                "valid_at_utc": fetched_at + timedelta(hours=step_k),
                "step_k": step_k,
                "dt_hours": 1.0,
                "t_out_c": 8.0,
                "gti_w_per_m2": 0.0,
                "gti_pv_w_per_m2": 0.0,
            }
            for step_k in range(4)
        ]
    )
    monkeypatch.setattr(HomeOptimizerAPI, "_get_repository", staticmethod(lambda: repository))

    original_base_request = api_service._base_request
    try:
        runtime_base_request = RunRequest.model_validate(
            {
                "horizon_hours": 4,
                "dhw_enabled": True,
                "dhw_C_top": 0.11628,
                "dhw_C_bot": 0.11628,
                "dhw_R_loss": 80.0,
            }
        )
        api_service.set_base_request(runtime_base_request)

        captured: dict[str, object] = {}

        def _fake_build_missing_overrides(*, request_data, repository, weather_rows, current_overrides=None):  # noqa: ANN001
            captured["request_data"] = dict(request_data)
            return {"dhw_v_tap_forecast": [0.0, 0.01, 0.0, 0.0]}

        monkeypatch.setattr(optimizer_module._FORECAST_SERVICE, "build_missing_overrides", _fake_build_missing_overrides)

        response = client.post(
            "/api/simulate",
            json={
                "dhw_enabled": True,
                "horizon_hours": 4,
            },
        )

        assert response.status_code == 200
        request_data = captured["request_data"]
        assert isinstance(request_data, dict)
        assert request_data["dhw_C_top"] == runtime_base_request.dhw_C_top
        assert request_data["dhw_C_bot"] == runtime_base_request.dhw_C_bot
    finally:
        api_service.set_base_request(original_base_request)


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

