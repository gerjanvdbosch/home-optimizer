from datetime import datetime
from typing import Protocol, TypeVar

import pandas as pd

from domain.models.config import AppConfig
from domain.models.dataset import DataDefinition, DatasetDefinition

T = TypeVar("T", bound=DataDefinition, contravariant=True)


class DataLoader(Protocol[T]):
    def supports(self, definition: DataDefinition) -> bool: ...

    def load(self, definition: T, start: datetime, end: datetime) -> pd.DataFrame: ...


class Forecaster(Protocol):
    @property
    def name(self) -> str: ...

    def dataset(self, config: AppConfig) -> DatasetDefinition: ...

    def fit(self, df: pd.DataFrame) -> None: ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def tune(self, df: pd.DataFrame) -> None: ...
