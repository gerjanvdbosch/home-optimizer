from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from home_optimizer.domain.series import NumericSeries, TextSeries


class DatasetSampleFrameReader(Protocol):
    def read_samples(
        self,
        *,
        interval_minutes: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame: ...


class DatasetSupportReader(Protocol):
    def read_forecast_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...

    def read_electricity_price_series(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        source: str,
        interval_minutes: int = 15,
    ) -> NumericSeries: ...
