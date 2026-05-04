from __future__ import annotations

from datetime import time
from typing import Annotated, Literal, Union

from pydantic import Field

from home_optimizer.domain.models import DomainModel



class DynamicPricing(DomainModel):
    mode: Literal["dynamic"] = "dynamic"
    delivery_area: str = "NL"
    currency: str = "EUR"


class FixedPricing(DomainModel):
    mode: Literal["fixed"] = "fixed"
    currency: str = "EUR"
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


ElectricityPricingConfig = Annotated[
    Union[DynamicPricing, FixedPricing],
    Field(discriminator="mode"),
]

