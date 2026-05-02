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

    def read_historical_weather_series(
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

    def latest(
        self,
        *,
        model_kind: str,
        model_name: str | None = None,
    ) -> IdentifiedModel | None: ...


class IdentifiedModelTrainer(Protocol):
    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel: ...


class MultiModelTrainer(Protocol):
    def train_all_models(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> list[IdentifiedModel]: ...

