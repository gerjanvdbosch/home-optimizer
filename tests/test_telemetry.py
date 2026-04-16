"""Tests for SQLAlchemy telemetry persistence and APScheduler collection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import ValidationError
from sqlalchemy import create_engine

from home_optimizer.sensors import LiveReadings, LocalBackend, SensorBackend
from home_optimizer.telemetry import (
    BufferedTelemetryCollector,
    TelemetryCollectorSettings,
    TelemetryRepository,
    aggregate_readings,
)


class SequenceBackend(SensorBackend):
    """Deterministic backend that returns a predefined sequence of snapshots."""

    def __init__(self, readings: list[LiveReadings]) -> None:
        self._readings = readings
        self._index = 0

    def read_all(self) -> LiveReadings:
        if self._index >= len(self._readings):
            raise RuntimeError("No more readings configured.")
        reading = self._readings[self._index]
        self._index += 1
        return reading

    def close(self) -> None:
        """Release backend resources.

        The deterministic test backend has no external resources.
        """


def _reading(timestamp: datetime, *, room_temperature_c: float, hp_mode: str) -> LiveReadings:
    """Create one fully populated telemetry snapshot for tests."""
    return LiveReadings(
        room_temperature_c=room_temperature_c,
        outdoor_temperature_c=8.0,
        hp_supply_temperature_c=31.0,
        hp_supply_target_temperature_c=33.0,
        hp_return_temperature_c=27.0,
        hp_flow_lpm=9.0,
        hp_electric_power_kw=2.0,
        hp_mode=hp_mode,
        p1_net_power_kw=1.4,
        pv_output_kw=0.6,
        thermostat_setpoint_c=20.5,
        dhw_top_temperature_c=52.0,
        dhw_bottom_temperature_c=45.0,
        shutter_living_room_pct=100.0,
        defrost_active=False,
        booster_heater_active=False,
        boiler_ambient_temp_c=18.0,
        refrigerant_condensation_temp_c=38.0,
        refrigerant_liquid_line_temp_c=28.0,
        discharge_temp_c=65.0,
        t_mains_estimated_c=10.5,
        timestamp=timestamp,
    )


def test_local_backend_reads_full_snapshot_from_json(tmp_path: Path) -> None:
    """The local backend must fail-fast unless the full telemetry snapshot is present."""
    sensor_file = tmp_path / "sensors.json"
    sensor_file.write_text(
        json.dumps(
            {
                "room_temperature_c": 20.5,
                "outdoor_temperature_c": 7.5,
                "hp_supply_temperature_c": 32.0,
                "hp_supply_target_temperature_c": 34.0,
                "hp_return_temperature_c": 28.0,
                "hp_flow_lpm": 8.5,
                "hp_electric_power_kw": 1.8,
                "hp_mode": "ufh",
                "p1_net_power_kw": 1.2,
                "pv_output_kw": 0.7,
                "thermostat_setpoint_c": 20.5,
                "dhw_top_temperature_c": 53.0,
                "dhw_bottom_temperature_c": 46.0,
                "shutter_living_room_pct": 80.0,
                "defrost_active": 0,
                "booster_heater_active": 0,
                "boiler_ambient_temp_c": 18.5,
                "refrigerant_condensation_temp_c": 37.5,
                "refrigerant_liquid_line_temp_c": 27.5,
                "discharge_temp_c": 63.0,
                "t_mains_estimated_c": 10.5,
            }
        ),
        encoding="utf-8",
    )

    backend = LocalBackend.from_json_file(sensor_file)
    reading = backend.read_all()

    assert reading.room_temperature_c == pytest.approx(20.5)
    assert reading.outdoor_temperature_c == pytest.approx(7.5)
    assert reading.hp_supply_target_temperature_c == pytest.approx(34.0)
    assert reading.hp_mode == "ufh"
    assert reading.hp_flow_lpm == pytest.approx(8.5)
    assert reading.shutter_living_room_pct == pytest.approx(80.0)
    assert reading.shutter_fraction == pytest.approx(0.8)
    assert reading.defrost_active is False
    assert reading.booster_heater_active is False
    assert reading.boiler_ambient_temp_c == pytest.approx(18.5)
    assert reading.refrigerant_condensation_temp_c == pytest.approx(37.5)
    assert reading.refrigerant_liquid_line_temp_c == pytest.approx(27.5)
    assert reading.discharge_temp_c == pytest.approx(63.0)


def test_aggregate_readings_computes_mean_and_last() -> None:
    """Aggregation must preserve both mean values and the last operating point."""
    t0 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    samples = [
        _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
        _reading(t0 + timedelta(seconds=30), room_temperature_c=20.5, hp_mode="ufh"),
        _reading(t0 + timedelta(seconds=60), room_temperature_c=21.0, hp_mode="ufh"),
    ]

    aggregate = aggregate_readings(samples)

    assert aggregate["bucket_start_utc"] == t0
    assert aggregate["bucket_end_utc"] == t0 + timedelta(seconds=60)
    assert aggregate["sample_count"] == 3
    assert aggregate["room_temperature_mean_c"] == pytest.approx(20.5)
    assert aggregate["room_temperature_last_c"] == pytest.approx(21.0)
    assert aggregate["hp_mode_last"] == "ufh"


def test_aggregate_readings_rejects_mixed_mode_buckets() -> None:
    """A single persisted bucket must never average UFH and DHW hydraulics together."""
    t0 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    samples = [
        _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
        _reading(t0 + timedelta(seconds=30), room_temperature_c=20.5, hp_mode="dhw"),
    ]

    with pytest.raises(ValueError, match="exactly one hp_mode"):
        aggregate_readings(samples)


def test_repository_and_collector_flush_round_trip(tmp_path: Path) -> None:
    """A flush must persist one aggregated telemetry bucket to SQLite."""
    t0 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    backend = SequenceBackend(
        [
            _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
            _reading(t0 + timedelta(seconds=30), room_temperature_c=20.4, hp_mode="ufh"),
        ]
    )
    database_url = f"sqlite:///{tmp_path / 'database.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    settings = TelemetryCollectorSettings(
        database_url=database_url,
        sampling_interval_seconds=30,
        flush_interval_seconds=300,
    )
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=settings,
    )

    repository.create_schema()
    collector.sample_once()
    collector.sample_once()
    collector.flush_once()

    rows = repository.list_aggregates()
    assert len(rows) == 1
    assert rows[0].sample_count == 2
    assert rows[0].room_temperature_mean_c == pytest.approx(20.2)
    assert rows[0].room_temperature_last_c == pytest.approx(20.4)
    assert rows[0].hp_mode_last == "ufh"


def test_collector_flushes_when_hp_mode_changes(tmp_path: Path) -> None:
    """The collector must split buckets immediately when the heat pump mode changes."""
    t0 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    backend = SequenceBackend(
        [
            _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
            _reading(t0 + timedelta(seconds=30), room_temperature_c=20.4, hp_mode="ufh"),
            _reading(t0 + timedelta(seconds=60), room_temperature_c=20.8, hp_mode="dhw"),
        ]
    )
    database_url = f"sqlite:///{tmp_path / 'mode-split.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    settings = TelemetryCollectorSettings(
        database_url=database_url,
        sampling_interval_seconds=30,
        flush_interval_seconds=300,
    )
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=settings,
    )

    repository.create_schema()
    collector.sample_once()
    collector.sample_once()
    collector.sample_once()
    collector.flush_once()

    rows = repository.list_aggregates()
    assert len(rows) == 2
    assert rows[0].sample_count == 2
    assert rows[0].hp_mode_last == "ufh"
    assert rows[1].sample_count == 1
    assert rows[1].hp_mode_last == "dhw"


def test_collector_start_registers_scheduler_jobs(tmp_path: Path) -> None:
    """Starting the collector must register both APScheduler interval jobs."""
    t0 = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    backend = SequenceBackend([_reading(t0, room_temperature_c=20.0, hp_mode="ufh")])
    engine = create_engine(f"sqlite:///{tmp_path / 'scheduler.sqlite3'}", future=True)
    repository = TelemetryRepository(engine=engine)
    settings = TelemetryCollectorSettings(
        database_url="sqlite:///unused.sqlite3",
        sampling_interval_seconds=30,
        flush_interval_seconds=300,
    )
    scheduler = BackgroundScheduler(timezone=settings.timezone_name)
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=settings,
        scheduler=scheduler,
    )

    collector.start()
    try:
        sample_job_id, flush_job_id = collector.job_ids()
        jobs = {job.id for job in collector.scheduler.get_jobs()}
        assert sample_job_id in jobs
        assert flush_job_id in jobs
    finally:
        collector.shutdown(flush=False)


def test_settings_fail_when_flush_interval_is_not_a_multiple() -> None:
    """Aggregation windows must contain a whole number of sample periods."""
    with pytest.raises(ValidationError):
        TelemetryCollectorSettings(
            database_url="sqlite:///database.sqlite3",
            sampling_interval_seconds=45,
            flush_interval_seconds=300,
        )


def test_aggregate_includes_new_sensor_fields() -> None:
    """Aggregation must include shutter, defrost, booster and refrigerant columns."""
    t0 = datetime(2026, 4, 15, 8, 0, tzinfo=timezone.utc)
    samples = [
        _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
        LiveReadings(
            room_temperature_c=20.2,
            outdoor_temperature_c=8.0,
            hp_supply_temperature_c=31.0,
            hp_supply_target_temperature_c=35.0,
            hp_return_temperature_c=27.0,
            hp_flow_lpm=9.0,
            hp_electric_power_kw=2.0,
            hp_mode="ufh",
            p1_net_power_kw=1.4,
            pv_output_kw=0.6,
            thermostat_setpoint_c=20.5,
            dhw_top_temperature_c=52.0,
            dhw_bottom_temperature_c=45.0,
            shutter_living_room_pct=50.0,
            defrost_active=True,
            booster_heater_active=False,
            boiler_ambient_temp_c=19.0,
            refrigerant_condensation_temp_c=40.0,
            refrigerant_liquid_line_temp_c=30.0,
            discharge_temp_c=68.0,
            t_mains_estimated_c=11.0,
            timestamp=t0 + timedelta(seconds=30),
        ),
    ]

    agg = aggregate_readings(samples)

    # Shutter: mean of [100, 50] = 75, last = 50
    assert agg["shutter_living_room_mean_pct"] == pytest.approx(75.0)
    assert agg["shutter_living_room_last_pct"] == pytest.approx(50.0)

    # Defrost: active in 1 of 2 samples → fraction = 0.5, last = True
    assert agg["defrost_active_fraction"] == pytest.approx(0.5)
    assert agg["defrost_active_last"] is True

    # Booster: never active → fraction = 0.0, last = False
    assert agg["booster_heater_active_fraction"] == pytest.approx(0.0)
    assert agg["booster_heater_active_last"] is False

    # Boiler ambient: mean of [18, 19] = 18.5, last = 19
    assert agg["boiler_ambient_temp_mean_c"] == pytest.approx(18.5)
    assert agg["boiler_ambient_temp_last_c"] == pytest.approx(19.0)

    # Refrigerant condensation: mean of [38, 40] = 39, last = 40
    assert agg["refrigerant_condensation_temp_mean_c"] == pytest.approx(39.0)
    assert agg["refrigerant_condensation_temp_last_c"] == pytest.approx(40.0)

    # Refrigerant liquid line: mean of [28, 30] = 29, last = 30
    assert agg["refrigerant_liquid_line_temp_mean_c"] == pytest.approx(29.0)
    assert agg["refrigerant_liquid_line_temp_last_c"] == pytest.approx(30.0)

    # Discharge temperature: mean of [65, 68] = 66.5, last = 68
    assert agg["discharge_temp_mean_c"] == pytest.approx(66.5)
    assert agg["discharge_temp_last_c"] == pytest.approx(68.0)

    # HP supply target: mean of [33, 35] = 34, last = 35
    assert agg["hp_supply_target_temperature_mean_c"] == pytest.approx(34.0)
    assert agg["hp_supply_target_temperature_last_c"] == pytest.approx(35.0)


def test_new_sensor_fields_persist_to_database(tmp_path: Path) -> None:
    """New sensor columns must round-trip through SQLite without data loss."""
    t0 = datetime(2026, 4, 15, 9, 0, tzinfo=timezone.utc)
    backend = SequenceBackend(
        [
            _reading(t0, room_temperature_c=20.0, hp_mode="ufh"),
            _reading(t0 + timedelta(seconds=30), room_temperature_c=20.4, hp_mode="ufh"),
        ]
    )
    database_url = f"sqlite:///{tmp_path / 'new_sensors.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    settings = TelemetryCollectorSettings(
        database_url=database_url,
        sampling_interval_seconds=30,
        flush_interval_seconds=300,
    )
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=settings,
    )

    repository.create_schema()
    collector.sample_once()
    collector.sample_once()
    collector.flush_once()

    rows = repository.list_aggregates()
    assert len(rows) == 1
    row = rows[0]

    # Shutter fully open in both samples → mean and last = 100.0
    assert row.shutter_living_room_mean_pct == pytest.approx(100.0)
    assert row.shutter_living_room_last_pct == pytest.approx(100.0)

    # No defrost or booster in default _reading fixture
    assert row.defrost_active_fraction == pytest.approx(0.0)
    assert row.defrost_active_last is False
    assert row.booster_heater_active_fraction == pytest.approx(0.0)
    assert row.booster_heater_active_last is False

    # Boiler ambient temperature
    assert row.boiler_ambient_temp_mean_c == pytest.approx(18.0)
    assert row.boiler_ambient_temp_last_c == pytest.approx(18.0)

    # Refrigerant temperatures
    assert row.refrigerant_condensation_temp_mean_c == pytest.approx(38.0)
    assert row.refrigerant_condensation_temp_last_c == pytest.approx(38.0)
    assert row.refrigerant_liquid_line_temp_mean_c == pytest.approx(28.0)
    assert row.refrigerant_liquid_line_temp_last_c == pytest.approx(28.0)
    assert row.discharge_temp_mean_c == pytest.approx(65.0)
    assert row.discharge_temp_last_c == pytest.approx(65.0)

    # HP supply target temperature
    assert row.hp_supply_target_temperature_mean_c == pytest.approx(33.0)
    assert row.hp_supply_target_temperature_last_c == pytest.approx(33.0)

    # Seasonal DHW parameter (t_mains_estimated_c)
    assert row.t_mains_estimated_mean_c == pytest.approx(10.5)
    assert row.t_mains_estimated_last_c == pytest.approx(10.5)

    # Derived quantities
    # hp_thermal_power = 9 L/min × 0.06 × 1.1628 × (31-27) ≈ 2.5076 kW
    assert row.hp_thermal_power_mean_kw == pytest.approx(9.0 * 0.06 * 1.1628 * 4.0, rel=1e-4)
    # household_elec = p1 + pv - hp_elec = 1.4 + 0.6 - 2.0 = 0.0
    assert row.household_elec_power_mean_kw == pytest.approx(0.0)


def test_forecast_persister_inserts_all_steps(tmp_path: Path) -> None:
    """ForecastPersister must persist all N forecast steps, skip duplicates."""
    from datetime import timezone
    from unittest.mock import MagicMock

    import numpy as np

    from home_optimizer.sensors.open_meteo import OpenMeteoClient, WeatherForecast
    from home_optimizer.telemetry import ForecastPersister

    _N = 4  # small horizon for speed

    # Build a minimal fake WeatherForecast with 4 steps
    valid_from = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    fake_forecast = WeatherForecast(
        outdoor_temperature_c=np.array([8.0, 7.5, 7.0, 6.5]),
        gti_w_per_m2=np.array([350.0, 200.0, 50.0, 0.0]),
        gti_pv_w_per_m2=np.array([280.0, 160.0, 40.0, 0.0]),
        horizon_steps=_N,
        dt_hours=1.0,
        valid_from=valid_from,
    )

    # Mock OpenMeteoClient so no real HTTP call is made
    mock_client = MagicMock(spec=OpenMeteoClient)
    mock_client.get_forecast.return_value = fake_forecast

    database_url = f"sqlite:///{tmp_path / 'forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    persister = ForecastPersister(mock_client, repository, horizon_hours=_N)

    # First persist: should insert 4 rows
    n_inserted = persister.persist_once()
    assert n_inserted == _N
    # get_forecast must have been called with the configured horizon
    mock_client.get_forecast.assert_called_once_with(horizon_hours=_N, dt_hours=1.0)

    rows = repository.list_forecast_snapshots()
    assert len(rows) == _N
    assert rows[0].step_k == 0
    # SQLite strips timezone info; compare the naive UTC value.
    assert rows[0].fetched_at_utc.replace(tzinfo=None) == valid_from.replace(tzinfo=None)
    assert rows[0].t_out_c == pytest.approx(8.0)
    assert rows[3].step_k == 3
    assert rows[3].gti_w_per_m2 == pytest.approx(0.0)

    # valid_at_utc for step 2 = valid_from + 2h
    expected_valid_at = valid_from + timedelta(hours=2)
    assert rows[2].valid_at_utc.replace(tzinfo=None) == expected_valid_at.replace(tzinfo=None)

    # Second persist with same forecast: all rows are duplicates → 0 inserted
    n_dupes = persister.persist_once()
    assert n_dupes == 0
    assert len(repository.list_forecast_snapshots()) == _N  # still 4, not 8
