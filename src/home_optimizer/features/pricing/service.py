from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.pricing import (
    DynamicPricing,
    ElectricityPricingConfig,
    FixedPricing,
    build_fixed_price_intervals,
    price_intervals_from_series,
)
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.pricing.ports import (
    DynamicElectricityPriceGatewayPort,
    ElectricityPriceRepositoryPort,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DYNAMIC_PRICE_SOURCE = "nordpool"
DEFAULT_FIXED_PRICE_SOURCE = "fixed_pricing"


class ElectricityPriceService:
    def __init__(
        self,
        pricing: ElectricityPricingConfig,
        repository: ElectricityPriceRepositoryPort,
        *,
        gateway: DynamicElectricityPriceGatewayPort | None = None,
        interval_minutes: int = 15,
        fixed_horizon_days: int = 1,
        dynamic_source: str = DEFAULT_DYNAMIC_PRICE_SOURCE,
        fixed_source: str = DEFAULT_FIXED_PRICE_SOURCE,
    ) -> None:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if fixed_horizon_days <= 0:
            raise ValueError("fixed_horizon_days must be greater than zero")

        self.pricing = pricing
        self.repository = repository
        self.gateway = gateway
        self.interval_minutes = interval_minutes
        self.fixed_horizon_days = fixed_horizon_days
        self.dynamic_source = dynamic_source
        self.fixed_source = fixed_source

    @property
    def enabled(self) -> bool:
        return not isinstance(self.pricing, DynamicPricing) or self.gateway is not None

    def refresh_prices(self, created_at: datetime | None = None) -> int:
        if not self.enabled:
            LOGGER.info("Electricity price refresh skipped: dynamic pricing gateway unavailable")
            return 0

        refresh_time = ensure_utc(created_at or utc_now())
        if isinstance(self.pricing, DynamicPricing):
            return self._refresh_dynamic_prices(self.pricing, refresh_time)
        return self._refresh_fixed_prices(self.pricing, refresh_time)

    def _refresh_dynamic_prices(self, pricing: DynamicPricing, refresh_time: datetime) -> int:
        assert self.gateway is not None

        written_rows = 0
        for delivery_date in self._dynamic_delivery_dates(refresh_time):
            series = self.gateway.fetch_day_ahead_prices(
                delivery_date=delivery_date,
                delivery_area=pricing.delivery_area,
                currency=pricing.currency,
            )
            intervals = price_intervals_from_series(
                series,
                source=self.dynamic_source,
                interval_minutes=self.interval_minutes,
            )
            written_rows += self.repository.upsert_intervals(intervals)

        LOGGER.info("Stored %s electricity price intervals for dynamic pricing", written_rows)
        return written_rows

    def _refresh_fixed_prices(self, pricing: FixedPricing, refresh_time: datetime) -> int:
        start_time = datetime.combine(refresh_time.date(), time.min, tzinfo=refresh_time.tzinfo)
        end_time = start_time + timedelta(days=self.fixed_horizon_days)
        intervals = build_fixed_price_intervals(
            pricing,
            start_time=start_time,
            end_time=end_time,
            source=self.fixed_source,
            interval_minutes=self.interval_minutes,
        )
        written_rows = self.repository.replace_future_intervals(
            source=self.fixed_source,
            from_time=start_time,
            intervals=intervals,
        )
        LOGGER.info("Stored %s electricity price intervals for fixed pricing", written_rows)
        return written_rows

    @staticmethod
    def _dynamic_delivery_dates(refresh_time: datetime) -> tuple[date, date]:
        today = refresh_time.date()
        return today, today + timedelta(days=1)


def electricity_price_refresh_interval_seconds(pricing: ElectricityPricingConfig) -> int:
    if isinstance(pricing, DynamicPricing):
        return 3600
    return 24 * 60 * 60




