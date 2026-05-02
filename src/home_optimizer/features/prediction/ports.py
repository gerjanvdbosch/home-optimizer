from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import IdentifiedModel, NumericSeries, TextSeries


class PredictionDataReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...

    def read_forecast_series(
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


class IdentifiedModelReader(Protocol):
    def latest(
        self,
        *,
        model_kind: str,
        model_name: str | None = None,
    ) -> IdentifiedModel | None: ...
