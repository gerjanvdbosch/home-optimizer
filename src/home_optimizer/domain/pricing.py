from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import ELECTRICITY_PRICE
from home_optimizer.domain.series import NumericPoint, NumericSeries
from home_optimizer.domain.time import normalize_utc_timestamp

DEFAULT_CURRENCY = "EUR"
DEFAULT_DELIVERY_AREA = "NL"


class DynamicPricing(DomainModel):
    mode: Literal["dynamic"] = "dynamic"
    delivery_area: str = DEFAULT_DELIVERY_AREA
    currency: str = DEFAULT_CURRENCY


class FixedPricing(DomainModel):
    mode: Literal["fixed"] = "fixed"
    currency: str = DEFAULT_CURRENCY
    peak_price: float = Field(gt=0)
    off_peak_price: float = Field(gt=0)
    feed_in_tariff: float = Field(ge=0)
    peak_start: time = time(7, 0)
    peak_end: time = time(23, 0)
    peak_monday: bool = True
    peak_tuesday: bool = True
    peak_wednesday: bool = True
    peak_thursday: bool = True
    peak_friday: bool = True
    peak_saturday: bool = False
    peak_sunday: bool = False

    @model_validator(mode="after")
    def _validate_peak_window(self) -> "FixedPricing":
        if self.peak_start == self.peak_end:
            raise ValueError("peak_start and peak_end must define a non-empty tariff window")
        return self

    @property
    def peak_days(self) -> frozenset[int]:
        mapping = {
            0: self.peak_monday,
            1: self.peak_tuesday,
            2: self.peak_wednesday,
            3: self.peak_thursday,
            4: self.peak_friday,
            5: self.peak_saturday,
            6: self.peak_sunday,
        }
        return frozenset(day for day, active in mapping.items() if active)

    def is_peak_time(self, timestamp: datetime) -> bool:
        local_time = timestamp.timetz() if timestamp.tzinfo else timestamp.time()
        candidate_time = local_time.replace(tzinfo=None)

        if self.peak_start < self.peak_end:
            return (
                timestamp.weekday() in self.peak_days
                and self.peak_start <= candidate_time < self.peak_end
            )

        active_day = timestamp.weekday() if candidate_time >= self.peak_start else (timestamp.weekday() - 1) % 7
        return active_day in self.peak_days and (
            candidate_time >= self.peak_start or candidate_time < self.peak_end
        )


ElectricityPricingConfig = Annotated[
    Union[DynamicPricing, FixedPricing],
    Field(discriminator="mode"),
]


def electricity_price_unit(currency: str = DEFAULT_CURRENCY) -> str:
    return f"{currency}/kWh"


def electricity_price_series(
    *,
    currency: str = DEFAULT_CURRENCY,
    points: Sequence[NumericPoint] = (),
) -> NumericSeries:
    return NumericSeries(
        name=ELECTRICITY_PRICE,
        unit=electricity_price_unit(currency),
        points=list(points),
    )


def empty_electricity_price_series(currency: str = DEFAULT_CURRENCY) -> NumericSeries:
    return electricity_price_series(currency=currency)


def build_daily_price_series(
    pricing: FixedPricing,
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int = 15,
) -> NumericSeries:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")
    if end_time <= start_time:
        return empty_electricity_price_series(pricing.currency)

    interval = timedelta(minutes=interval_minutes)
    points: list[NumericPoint] = []
    cursor = start_time

    while cursor < end_time:
        price = pricing.peak_price if pricing.is_peak_time(cursor) else pricing.off_peak_price
        points.append(NumericPoint(timestamp=normalize_utc_timestamp(cursor), value=price))
        cursor += interval

    return electricity_price_series(currency=pricing.currency, points=points)


def resolve_daily_price_series(
    pricing: ElectricityPricingConfig,
    *,
    start_time: datetime,
    end_time: datetime,
    fetched_series: NumericSeries | None = None,
    interval_minutes: int = 15,
) -> NumericSeries:
    if isinstance(pricing, DynamicPricing):
        if fetched_series is not None:
            return fetched_series
        return empty_electricity_price_series(pricing.currency)

    return build_daily_price_series(
        pricing,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=interval_minutes,
    )

