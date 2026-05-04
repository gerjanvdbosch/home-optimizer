from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.series import NumericSeries


class ElectricityPriceRepositoryPort(Protocol):
    def latest_interval_end(self, source: str) -> datetime | None: ...

    def upsert_intervals(self, intervals: list[PriceInterval]) -> int: ...

    def replace_future_intervals(
        self,
        *,
        source: str,
        from_time: datetime,
        intervals: list[PriceInterval],
    ) -> int: ...


class DynamicElectricityPriceGatewayPort(Protocol):
    def fetch_day_ahead_prices(
        self,
        *,
        delivery_date: date,
        delivery_area: str,
        currency: str,
        market: str = "DayAhead",
    ) -> NumericSeries: ...



