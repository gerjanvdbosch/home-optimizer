from __future__ import annotations

from datetime import date, datetime, timezone

from home_optimizer.domain import ELECTRICITY_PRICE, NumericPoint, NumericSeries
from home_optimizer.domain.pricing import DynamicPricing, FixedPricing
from home_optimizer.features.pricing.service import ElectricityPriceService


class FakePriceRepository:
    def __init__(self) -> None:
        self.upserted = []
        self.replaced = []

    def latest_interval_end(self, source: str):
        return None

    def upsert_intervals(self, intervals):
        self.upserted.append(intervals)
        return len(intervals)

    def replace_future_intervals(self, *, source, from_time, intervals):
        self.replaced.append((source, from_time, intervals))
        return len(intervals)


class FakeDynamicGateway:
    def __init__(self, responses: dict[date, NumericSeries]) -> None:
        self.responses = responses
        self.calls: list[date] = []

    def fetch_day_ahead_prices(self, *, delivery_date, delivery_area, currency, market="DayAhead"):
        self.calls.append(delivery_date)
        return self.responses[delivery_date]


def test_electricity_price_service_stores_known_dynamic_prices_for_today_and_tomorrow() -> None:
    repository = FakePriceRepository()
    gateway = FakeDynamicGateway(
        {
            date(2026, 5, 4): NumericSeries(
                name=ELECTRICITY_PRICE,
                unit="EUR/kWh",
                points=[
                    NumericPoint(timestamp="2026-05-04T00:00:00+00:00", value=0.21),
                    NumericPoint(timestamp="2026-05-04T00:15:00+00:00", value=0.22),
                ],
            ),
            date(2026, 5, 5): NumericSeries(
                name=ELECTRICITY_PRICE,
                unit="EUR/kWh",
                points=[
                    NumericPoint(timestamp="2026-05-05T00:00:00+00:00", value=0.31),
                ],
            ),
        }
    )
    service = ElectricityPriceService(DynamicPricing(), repository, gateway=gateway)

    written_rows = service.refresh_prices(datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc))

    assert gateway.calls == [date(2026, 5, 4), date(2026, 5, 5)]
    assert written_rows == 3
    assert len(repository.upserted) == 2
    assert [interval.value for intervals in repository.upserted for interval in intervals] == [0.21, 0.22, 0.31]


def test_electricity_price_service_replaces_future_fixed_prices_with_compressed_intervals() -> None:
    repository = FakePriceRepository()
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    service = ElectricityPriceService(pricing, repository, fixed_horizon_days=2)

    written_rows = service.refresh_prices(datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc))

    assert written_rows == 5
    assert repository.replaced[0][0] == "fixed_pricing"
    assert repository.replaced[0][1] == datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    intervals = repository.replaced[0][2]
    assert len(intervals) == 5
    assert intervals[0].start_time_utc == datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    assert intervals[0].end_time_utc == datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc)
    assert intervals[1].value == 0.32
    assert intervals[2].start_time_utc == datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc)
    assert intervals[2].end_time_utc == datetime(2026, 5, 5, 7, 0, tzinfo=timezone.utc)

