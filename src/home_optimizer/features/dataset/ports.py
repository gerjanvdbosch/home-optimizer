from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class DatasetSampleFrameReader(Protocol):
    def read_samples(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame: ...

    def read_forecast_values(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        created_at_start_time: datetime | None = None,
        created_at_end_time: datetime | None = None,
    ) -> pd.DataFrame: ...

    def read_electricity_price_intervals(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> pd.DataFrame: ...
