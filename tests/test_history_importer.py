from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.bootstrap.settings import AppSettings
from home_optimizer.features.history_import.repository import HistoryImportRepository
from home_optimizer.features.history_import.schemas import HistoryImportRequest
from home_optimizer.features.history_import.service import HistoryImportService
from home_optimizer.shared.db.orm_models import ImportChunk, Sample1m
from home_optimizer.shared.db.session import Database
from home_optimizer.shared.sensors.definitions import SensorSpec


class FakeHomeAssistantClient:
    def __init__(self, history: list[dict[str, str]]) -> None:
        self.history = history
        self.calls = 0

    def get_history(self, **kwargs) -> list[dict[str, str]]:
        self.calls += 1
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        rows: list[dict[str, str]] = []

        for item in self.history:
            ts_raw = item.get("last_changed") or item.get("last_updated")
            if ts_raw is None:
                continue

            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))

            if start_time is not None and ts < start_time:
                continue

            if end_time is not None and ts >= end_time:
                continue

            rows.append(item)

        return rows


def test_mean_import_converts_values_and_skips_imported_chunk(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    ha = FakeHomeAssistantClient(
        [
            {
                "state": "1000",
                "last_changed": "2026-04-14T00:00:10+00:00",
            },
            {
                "state": "2000",
                "last_changed": "2026-04-14T00:00:40+00:00",
            },
            {
                "state": "unknown",
                "last_changed": "2026-04-14T00:01:00+00:00",
            },
        ]
    )
    spec = SensorSpec(
        name="power",
        entity_id="sensor.power",
        category="energy",
        unit="kW",
        method="mean",
        conversion_factor=0.001,
    )
    importer = HistoryImportService(ha, HistoryImportRepository(db), chunk_days=1)
    start = datetime(2026, 4, 14, tzinfo=timezone.utc)
    end = datetime(2026, 4, 15, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 1
    assert importer.import_sensor(spec, start, end) == 0
    assert ha.calls == 1

    with db.session() as session:
        row = session.query(Sample1m).one()

    assert row.timestamp_minute_utc == "2026-04-14T00:00:00+00:00"
    assert row.mean_real == 1.5
    assert row.min_real == 1.0
    assert row.max_real == 2.0
    assert row.last_real == 2.0
    assert row.sample_count == 2


def test_forward_fill_creates_one_row_per_minute(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    ha = FakeHomeAssistantClient(
        [
            {
                "state": "off",
                "last_changed": "2026-04-14T00:00:00+00:00",
            },
            {
                "state": "on",
                "last_changed": "2026-04-14T00:02:00+00:00",
            },
        ]
    )
    spec = SensorSpec(
        name="switch",
        entity_id="binary_sensor.switch",
        category="heatpump",
        unit="bool",
        method="ffill",
    )
    importer = HistoryImportService(ha, HistoryImportRepository(db), chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 4, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 4

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.last_bool for row in rows] == [0, 0, 1, 1]
    assert [row.sample_count for row in rows] == [1, 1, 1, 1]


def test_forward_fill_text_mode_keeps_off_as_text(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    ha = FakeHomeAssistantClient(
        [
            {
                "state": "off",
                "last_changed": "2026-04-14T00:00:00+00:00",
            },
            {
                "state": "heat",
                "last_changed": "2026-04-14T00:02:00+00:00",
            },
        ]
    )
    spec = SensorSpec(
        name="hp_mode",
        entity_id="sensor.warmtepomp_mode",
        category="heatpump",
        unit=None,
        method="ffill",
    )
    importer = HistoryImportService(ha, HistoryImportRepository(db), chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 4, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 4

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.last_text for row in rows] == ["off", "off", "heat", "heat"]
    assert [row.last_bool for row in rows] == [None, None, None, None]
    assert [row.sample_count for row in rows] == [1, 1, 1, 1]


def test_time_weighted_mean_creates_one_row_per_minute(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    ha = FakeHomeAssistantClient(
        [
            {
                "state": "10",
                "last_changed": "2026-04-14T00:00:00+00:00",
            },
            {
                "state": "20",
                "last_changed": "2026-04-14T00:02:30+00:00",
            },
        ]
    )
    spec = SensorSpec(
        name="hp_flow",
        entity_id="sensor.flow",
        category="heatpump",
        unit="Lmin",
        method="time_weighted_mean",
    )
    importer = HistoryImportService(ha, HistoryImportRepository(db), chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 4, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 4

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.timestamp_minute_utc for row in rows] == [
        "2026-04-14T00:00:00+00:00",
        "2026-04-14T00:01:00+00:00",
        "2026-04-14T00:02:00+00:00",
        "2026-04-14T00:03:00+00:00",
    ]
    assert [row.mean_real for row in rows] == [10.0, 10.0, 15.0, 20.0]
    assert [row.last_real for row in rows] == [10.0, 10.0, 20.0, 20.0]
    assert [row.sample_count for row in rows] == [1, 1, 1, 1]


def test_time_weighted_mean_carries_value_across_chunks(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    ha = FakeHomeAssistantClient(
        [
            {
                "state": "10",
                "last_changed": "2026-04-14T00:00:00+00:00",
            },
            {
                "state": "20",
                "last_changed": "2026-04-14T00:03:00+00:00",
            },
        ]
    )
    spec = SensorSpec(
        name="hp_flow",
        entity_id="sensor.flow",
        category="heatpump",
        unit="Lmin",
        method="time_weighted_mean",
    )
    importer = HistoryImportService(ha, HistoryImportRepository(db), chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    split = datetime(2026, 4, 14, 0, 2, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 5, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, split) == 2
    assert importer.import_sensor(spec, split, end) == 3

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.timestamp_minute_utc for row in rows] == [
        "2026-04-14T00:00:00+00:00",
        "2026-04-14T00:01:00+00:00",
        "2026-04-14T00:02:00+00:00",
        "2026-04-14T00:03:00+00:00",
        "2026-04-14T00:04:00+00:00",
    ]
    assert [row.mean_real for row in rows] == [10.0, 10.0, 10.0, 20.0, 20.0]
    assert [row.sample_count for row in rows] == [1, 1, 1, 1, 1]


def test_import_chunk_timestamps_are_stored_without_microseconds(tmp_path) -> None:
    db = Database(str(tmp_path / "history.db"))
    db.init_schema()
    repository = HistoryImportRepository(db)
    spec = SensorSpec(
        name="power",
        entity_id="sensor.power",
        category="energy",
        unit="kW",
        method="mean",
    )
    start = datetime(2026, 4, 14, 0, 0, 0, 123456, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 1, 2, 3, 654321, tzinfo=timezone.utc)

    repository.mark_chunk_imported(spec, start, end, row_count=7)

    with db.session() as session:
        chunk = session.query(ImportChunk).one()

    assert chunk.start_time_utc == "2026-04-14T00:00:00+00:00"
    assert chunk.end_time_utc == "2026-04-14T01:02:03+00:00"
    assert chunk.imported_at_utc.endswith("+00:00")
    assert "." not in chunk.imported_at_utc


def test_history_import_request_uses_max_days_back_when_configured() -> None:
    settings = AppSettings(
        history_import_max_days_back=10,
    )

    request = HistoryImportRequest.from_settings(settings, specs=[])

    assert request.end_time.tzinfo == timezone.utc
    assert request.start_time.tzinfo == timezone.utc
    assert request.end_time - request.start_time == timedelta(days=10)
