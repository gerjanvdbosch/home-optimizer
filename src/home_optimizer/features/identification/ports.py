from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import IdentifiedModel, NumericSeries, TextSeries


class IdentificationDataReader(Protocol):
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


class IdentifiedModelRepository(Protocol):
    def save(self, model: IdentifiedModel) -> None: ...

    def latest(self, *, model_kind: str) -> IdentifiedModel | None: ...
