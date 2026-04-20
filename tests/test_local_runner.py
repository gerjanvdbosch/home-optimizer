"""Tests for the local development runner."""

from __future__ import annotations

import json
from pathlib import Path

from home_optimizer import local_runner


def _write_sensors_json(path: Path) -> None:
    """Write one complete local-runner telemetry snapshot."""
    path.write_text(
        json.dumps(
            {
                "room_temperature_c": 20.5,
                "outdoor_temperature_c": 8.0,
                "hp_supply_temperature_c": 31.0,
                "hp_supply_target_temperature_c": 33.0,
                "hp_return_temperature_c": 27.0,
                "hp_flow_lpm": 9.0,
                "hp_electric_power_kw": 2.0,
                "hp_mode": "ufh",
                "p1_net_power_kw": 1.4,
                "pv_output_kw": 0.6,
                "thermostat_setpoint_c": 20.5,
                "dhw_top_temperature_c": 52.0,
                "dhw_bottom_temperature_c": 45.0,
                "shutter_living_room_pct": 100.0,
                "defrost_active": 0,
                "booster_heater_active": 0,
                "boiler_ambient_temp_c": 18.0,
                "refrigerant_condensation_temp_c": 38.0,
                "refrigerant_liquid_line_temp_c": 28.0,
                "discharge_temp_c": 65.0,
                "t_mains_estimated_c": 10.5,
                "pv_total_kwh": 1000.0,
                "hp_electric_total_kwh": 500.0,
                "p1_import_total_kwh": 800.0,
                "p1_export_total_kwh": 200.0,
            }
        ),
        encoding="utf-8",
    )


def test_parse_args_exposes_calibration_flags() -> None:
    """Local runner CLI must expose automatic calibration controls."""
    args = local_runner._parse_args(
        [
            "--calibration-interval",
            "900",
            "--calibration-min-history-hours",
            "12",
        ]
    )

    assert args.calibration_interval == 900
    assert args.calibration_min_history_hours == 12.0


def test_parse_args_exposes_forecast_training_flags() -> None:
    """Local runner CLI must expose nightly persisted forecast-model training controls."""
    args = local_runner._parse_args(
        [
            "--forecast-training-hour-utc",
            "3",
            "--forecast-training-minute-utc",
            "45",
            "--no-forecast-training-enabled",
        ]
    )

    assert args.forecast_training_enabled is False
    assert args.forecast_training_hour_utc == 3
    assert args.forecast_training_minute_utc == 45


def test_main_runs_initial_automatic_calibration_when_enabled(tmp_path, monkeypatch) -> None:
    """Local runner must trigger automatic calibration during startup."""
    sensors_path = tmp_path / "sensors.json"
    _write_sensors_json(sensors_path)
    database_path = tmp_path / "local-runner.sqlite3"

    calibration_calls: list[tuple[str, float]] = []

    class FakeForecastPersister:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
            pass

        def persist_once(self) -> int:
            return 0

        def start(self, scheduler, *, run_immediately: bool = True) -> None:  # noqa: ANN001
            return None

    monkeypatch.setattr(local_runner, "ForecastPersister", FakeForecastPersister)
    monkeypatch.setattr(local_runner, "OpenMeteoClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(local_runner.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(local_runner.TelemetryRepository, "export_to_env", lambda self: None)

    def fake_run_and_persist_automatic_calibration(repository, *, base_request, settings):  # noqa: ANN001
        calibration_calls.append((repository.engine.url.render_as_string(hide_password=False), settings.min_history_hours))
        return None

    monkeypatch.setattr(
        local_runner,
        "run_and_persist_automatic_calibration",
        fake_run_and_persist_automatic_calibration,
    )

    local_runner.main(
        [
            "--database",
            str(database_path),
            "--sensors-json",
            str(sensors_path),
            "--mpc-interval",
            "0",
            "--calibration-interval",
            "60",
            "--calibration-min-history-hours",
            "12",
        ]
    )

    assert len(calibration_calls) == 1
    database_url, min_history_hours = calibration_calls[0]
    assert str(database_path.resolve()) in database_url
    assert min_history_hours == 12.0


def test_main_runs_initial_forecast_model_training_when_enabled(tmp_path, monkeypatch) -> None:
    """Local runner must train persisted forecast models once during startup when enabled."""
    sensors_path = tmp_path / "sensors.json"
    _write_sensors_json(sensors_path)
    database_path = tmp_path / "local-runner-forecast.sqlite3"

    forecast_training_calls: list[str] = []

    class FakeForecastPersister:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
            pass

        def persist_once(self) -> int:
            return 0

        def start(self, scheduler, *, run_immediately: bool = True) -> None:  # noqa: ANN001
            return None

    class FakeForecastService:
        def train_and_persist_models(self, *, repository, base_request_data=None):  # noqa: ANN001
            forecast_training_calls.append(repository.engine.url.render_as_string(hide_password=False))
            assert base_request_data is not None
            return {"shutter_forecast": None}

    monkeypatch.setattr(local_runner, "ForecastPersister", FakeForecastPersister)
    monkeypatch.setattr(local_runner, "ForecastService", FakeForecastService)
    monkeypatch.setattr(local_runner, "OpenMeteoClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(local_runner.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(local_runner.TelemetryRepository, "export_to_env", lambda self: None)

    local_runner.main(
        [
            "--database",
            str(database_path),
            "--sensors-json",
            str(sensors_path),
            "--mpc-interval",
            "0",
            "--calibration-interval",
            "0",
            "--forecast-training-hour-utc",
            "1",
            "--forecast-training-minute-utc",
            "30",
        ]
    )

    assert len(forecast_training_calls) == 1
    assert str(database_path.resolve()) in forecast_training_calls[0]


