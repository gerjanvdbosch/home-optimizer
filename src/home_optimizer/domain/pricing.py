from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.names import ELECTRICITY_PRICE
from home_optimizer.domain.series import NumericPoint, NumericSeries
from home_optimizer.domain.time import ensure_utc, normalize_utc_timestamp, parse_datetime

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


class PriceInterval(DomainModel):
    start_time_utc: datetime
    end_time_utc: datetime
    source: str
    value: float = Field(ge=0)
    name: str = ELECTRICITY_PRICE
    unit: str = f"{DEFAULT_CURRENCY}/kWh"

    @model_validator(mode="after")
    def _validate_interval(self) -> "PriceInterval":
        start_time_utc = ensure_utc(self.start_time_utc)
        end_time_utc = ensure_utc(self.end_time_utc)
        if end_time_utc <= start_time_utc:
            raise ValueError("price interval end_time_utc must be after start_time_utc")
        object.__setattr__(self, "start_time_utc", start_time_utc)
        object.__setattr__(self, "end_time_utc", end_time_utc)
        return self


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


def price_intervals_from_series(
    series: NumericSeries,
    *,
    source: str,
    interval_minutes: int = 15,
) -> list[PriceInterval]:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")
    if not series.points:
        return []

    ordered_points = sorted(series.points, key=lambda point: point.timestamp)
    default_interval = timedelta(minutes=interval_minutes)
    unit = series.unit or electricity_price_unit()
    intervals: list[PriceInterval] = []

    for index, point in enumerate(ordered_points):
        start_time = parse_datetime(point.timestamp)
        if index + 1 < len(ordered_points):
            end_time = parse_datetime(ordered_points[index + 1].timestamp)
        else:
            end_time = start_time + default_interval

        if end_time <= start_time:
            continue

        interval = PriceInterval(
            start_time_utc=start_time,
            end_time_utc=end_time,
            source=source,
            name=series.name,
            unit=unit,
            value=float(point.value),
        )

        if (
            intervals
            and intervals[-1].name == interval.name
            and intervals[-1].unit == interval.unit
            and intervals[-1].source == interval.source
            and intervals[-1].value == interval.value
            and intervals[-1].end_time_utc == interval.start_time_utc
        ):
            intervals[-1] = intervals[-1].model_copy(update={"end_time_utc": interval.end_time_utc})
            continue

        intervals.append(interval)

    return intervals


def build_fixed_price_intervals(
    pricing: FixedPricing,
    *,
    start_time: datetime,
    end_time: datetime,
    source: str,
    interval_minutes: int = 15,
) -> list[PriceInterval]:
    series = build_daily_price_series(
        pricing,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=interval_minutes,
    )
    return price_intervals_from_series(
        series,
        source=source,
        interval_minutes=interval_minutes,
    )


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

