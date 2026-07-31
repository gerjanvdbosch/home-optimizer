from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

import pandas as pd
from optuna import Study

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DataDefinition, DatasetDefinition
from domain.models.state import BacktestResult

JsonType = dict[str, Any] | list[Any]


T = TypeVar("T", bound=DataDefinition, contravariant=True)


class DataLoader(Protocol[T]):
    def supports(self, definition: DataDefinition) -> bool: ...

    def load(self, definition: T, start: datetime, end: datetime) -> pd.DataFrame: ...


class Forecaster(Protocol):
    @property
    def name(self) -> ForecasterType: ...

    @property
    def target_column(self) -> str: ...

    @property
    def exog_columns(self) -> list[str]: ...

    @property
    def y_axis(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def dataset(self, config: Config) -> DatasetDefinition: ...

    def fit(self, df: pd.DataFrame): ...

    def predict(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
        steps: int = 24,
    ) -> pd.Series: ...

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult: ...

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
