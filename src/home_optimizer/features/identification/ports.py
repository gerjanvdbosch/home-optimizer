from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain.charts import ChartSeries


class IdentificationDataReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartSeries]: ...

    def read_forecast_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartSeries]: ...
