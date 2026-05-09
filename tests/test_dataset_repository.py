from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import insert

from home_optimizer.infrastructure.database.dataset_repository import DatasetRepository
from home_optimizer.infrastructure.database.orm_models import (
    ElectricityPriceIntervalValue,
    ForecastValue,
    Sample1m,
    Sample15m,
)
from home_optimizer.infrastructure.database.session import Database


def test_dataset_repository_reads_raw_samples_1m_and_15m_as_dataframes(tmp_path) -> None:
    database = Database(str(tmp_path / "dataset_repository.sqlite"))
    database.init_schema()
    repository = DatasetRepository(database)

    with database.session() as session:
        session.execute(
            insert(Sample1m),
            [
                {
                    "timestamp_minute_utc": "2026-02-08T00:00:00Z",
                    "name": "room_temperature",
                    "source": "ha",
                    "entity_id": "sensor.room_temperature",
                    "category": "measurement",
                    "unit": "°C",
                    "mean_real": 20.1,
                    "min_real": 20.0,
                    "max_real": 20.2,
                    "last_real": 20.2,
                    "last_text": None,
                    "last_bool": None,
                    "sample_count": 1,
                }
            ],
        )
        session.execute(
            insert(Sample15m),
            [
                {
                    "timestamp_15m_utc": "2026-02-08T00:15:00Z",
                    "name": "hp_mode",
                    "source": "ha",
                    "entity_id": "sensor.hp_mode",
                    "category": "measurement",
                    "unit": None,
                    "mean_real": None,
                    "min_real": None,
                    "max_real": None,
                    "last_real": None,
                    "last_text": "space_heating",
                    "last_bool": None,
                    "sample_count": 1,
                }
            ],
        )
        session.commit()

    frame_1m = repository.read_samples_1m(
        start_time=datetime(2026, 2, 8, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 8, 0, 30, tzinfo=timezone.utc),
    )
    frame_15m = repository.read_samples_15m(
        start_time=datetime(2026, 2, 8, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 8, 0, 30, tzinfo=timezone.utc),
    )

    assert list(frame_1m.columns) == [
        "timestamp_minute_utc",
        "name",
        "source",
        "entity_id",
        "category",
        "unit",
        "mean_real",
        "min_real",
        "max_real",
        "last_real",
        "last_text",
        "last_bool",
        "sample_count",
    ]
    assert frame_1m.shape == (1, 13)
    assert frame_1m.iloc[0]["name"] == "room_temperature"
    assert frame_1m.iloc[0]["mean_real"] == 20.1

    assert "timestamp_15m_utc" in frame_15m.columns
    assert frame_15m.shape == (1, 13)
    assert frame_15m.iloc[0]["name"] == "hp_mode"
    assert frame_15m.iloc[0]["last_text"] == "space_heating"


def test_dataset_repository_falls_back_to_15m_when_1m_is_empty(tmp_path) -> None:
    database = Database(str(tmp_path / "dataset_repository_fallback.sqlite"))
    database.init_schema()
    repository = DatasetRepository(database)

    with database.session() as session:
        session.execute(
            insert(Sample15m),
            [
                {
                    "timestamp_15m_utc": "2026-02-08T00:15:00Z",
                    "name": "room_temperature",
                    "source": "ha",
                    "entity_id": "sensor.room_temperature",
                    "category": "measurement",
                    "unit": "°C",
                    "mean_real": 20.5,
                    "min_real": 20.4,
                    "max_real": 20.6,
                    "last_real": 20.6,
                    "last_text": None,
                    "last_bool": None,
                    "sample_count": 1,
                }
            ],
        )
        session.commit()

    frame = repository.read_samples(
        start_time=datetime(2026, 2, 8, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 8, 0, 30, tzinfo=timezone.utc),
    )

    assert "timestamp_utc" in frame.columns
    assert frame.shape[0] == 1
    assert frame.iloc[0]["name"] == "room_temperature"
    assert frame.iloc[0]["mean_real"] == 20.5


def test_dataset_repository_reads_raw_forecast_values_as_dataframe(tmp_path) -> None:
    database = Database(str(tmp_path / "dataset_repository_forecast.sqlite"))
    database.init_schema()
    repository = DatasetRepository(database)

    with database.session() as session:
        session.execute(
            insert(ForecastValue),
            [
                {
                    "created_at_utc": "2026-02-08T00:00:00Z",
                    "forecast_time_utc": "2026-02-08T01:00:00Z",
                    "name": "temperature",
                    "source": "openmeteo",
                    "unit": "°C",
                    "value": 5.5,
                }
            ],
        )
        session.commit()

    frame = repository.read_forecast_values(
        start_time=datetime(2026, 2, 8, 0, 30, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 8, 1, 30, tzinfo=timezone.utc),
        names=["temperature"],
        sources=["openmeteo"],
    )

    assert frame.shape == (1, 6)
    assert list(frame.columns) == [
        "created_at_utc",
        "forecast_time_utc",
        "name",
        "source",
        "unit",
        "value",
    ]
    assert frame.iloc[0]["value"] == 5.5


def test_dataset_repository_reads_overlapping_price_intervals_as_dataframe(tmp_path) -> None:
    database = Database(str(tmp_path / "dataset_repository_prices.sqlite"))
    database.init_schema()
    repository = DatasetRepository(database)

    with database.session() as session:
        session.execute(
            insert(ElectricityPriceIntervalValue),
            [
                {
                    "name": "electricity_price",
                    "start_time_utc": "2026-02-08T00:00:00Z",
                    "end_time_utc": "2026-02-08T01:00:00Z",
                    "source": "nordpool",
                    "unit": "EUR/kWh",
                    "value": 0.25,
                },
                {
                    "name": "electricity_price",
                    "start_time_utc": "2026-02-08T01:00:00Z",
                    "end_time_utc": "2026-02-08T02:00:00Z",
                    "source": "nordpool",
                    "unit": "EUR/kWh",
                    "value": 0.30,
                },
            ],
        )
        session.commit()

    frame = repository.read_electricity_price_intervals(
        start_time=datetime(2026, 2, 8, 0, 30, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 8, 1, 30, tzinfo=timezone.utc),
        names=["electricity_price"],
        sources=["nordpool"],
    )

    assert frame.shape == (2, 6)
    assert list(frame.columns) == [
        "name",
        "start_time_utc",
        "end_time_utc",
        "source",
        "unit",
        "value",
    ]
    assert list(frame["value"]) == [0.25, 0.30]
