from datetime import datetime
from pathlib import Path
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

    def fit(self, df: pd.DataFrame): ...

    def predict(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
        steps: int = 24,
    ) -> pd.Series: ...

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
    ) -> tuple[pd.DataFrame, object]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
