from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text

from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.infrastructure.database.electricity_price_repository import ElectricityPriceRepository
from home_optimizer.infrastructure.database.session import Database


def test_electricity_price_repository_upserts_intervals(tmp_path) -> None:
    database = Database(str(tmp_path / "prices.sqlite"))
    database.init_schema()
    repository = ElectricityPriceRepository(database)

    written_rows = repository.upsert_intervals(
        [
            PriceInterval(
                start_time_utc=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 0, 15, tzinfo=timezone.utc),
                source="nordpool",
                unit="EUR/kWh",
                value=0.21,
            )
        ]
    )
    repository.upsert_intervals(
        [
            PriceInterval(
                start_time_utc=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 0, 15, tzinfo=timezone.utc),
                source="nordpool",
                unit="EUR/kWh",
                value=0.25,
            )
        ]
    )

    assert written_rows == 1
    with database.session() as session:
        rows = session.execute(
            text("SELECT start_time_utc, end_time_utc, source, unit, value FROM electricity_price_intervals")
        ).all()

    assert rows == [
        (
            "2026-05-04T00:00:00+00:00",
            "2026-05-04T00:15:00+00:00",
            "nordpool",
            "EUR/kWh",
            0.25,
        )
    ]


def test_electricity_price_repository_replace_future_intervals_removes_overlaps(tmp_path) -> None:
    database = Database(str(tmp_path / "prices.sqlite"))
    database.init_schema()
    repository = ElectricityPriceRepository(database)

    repository.upsert_intervals(
        [
            PriceInterval(
                start_time_utc=datetime(2026, 5, 3, 23, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc),
                source="fixed_pricing",
                unit="EUR/kWh",
                value=0.21,
            ),
            PriceInterval(
                start_time_utc=datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc),
                source="fixed_pricing",
                unit="EUR/kWh",
                value=0.32,
            ),
        ]
    )

    repository.replace_future_intervals(
        source="fixed_pricing",
        from_time=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        intervals=[
            PriceInterval(
                start_time_utc=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 6, 0, tzinfo=timezone.utc),
                source="fixed_pricing",
                unit="EUR/kWh",
                value=0.19,
            ),
            PriceInterval(
                start_time_utc=datetime(2026, 5, 4, 6, 0, tzinfo=timezone.utc),
                end_time_utc=datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc),
                source="fixed_pricing",
                unit="EUR/kWh",
                value=0.33,
            ),
        ],
    )

    with database.session() as session:
        rows = session.execute(
            text(
                "SELECT start_time_utc, end_time_utc, value FROM electricity_price_intervals ORDER BY start_time_utc"
            )
        ).all()

    assert rows == [
        ("2026-05-04T00:00:00+00:00", "2026-05-04T06:00:00+00:00", 0.19),
        ("2026-05-04T06:00:00+00:00", "2026-05-04T23:00:00+00:00", 0.33),
    ]

