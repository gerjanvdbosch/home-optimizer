from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain.series import NumericSeries, TextSeries


class DatasetDataReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[TextSeries]: ...

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
