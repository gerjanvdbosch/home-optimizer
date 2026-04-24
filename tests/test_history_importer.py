from __future__ import annotations

from datetime import datetime, timezone

from config.sensor_definitions import SensorSpec
from database.models import Sample1m
from database.session import Database
from importer.history_importer import HomeAssistantHistoryImporter


class FakeHomeAssistantClient:
    def __init__(self, history: list[dict[str, str]]) -> None:
        self.history = history
        self.calls = 0

    def get_history(self, **kwargs) -> list[dict[str, str]]:
        self.calls += 1
        return self.history


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
    importer = HomeAssistantHistoryImporter(ha, db, chunk_days=1)
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
    importer = HomeAssistantHistoryImporter(ha, db, chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 4, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 4

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.last_bool for row in rows] == [0, 0, 1, 1]


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
    importer = HomeAssistantHistoryImporter(ha, db, chunk_days=1)
    start = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 14, 0, 4, tzinfo=timezone.utc)

    assert importer.import_sensor(spec, start, end) == 4

    with db.session() as session:
        rows = session.query(Sample1m).order_by(Sample1m.timestamp_minute_utc).all()

    assert [row.last_text for row in rows] == ["off", "off", "heat", "heat"]
    assert [row.last_bool for row in rows] == [None, None, None, None]
