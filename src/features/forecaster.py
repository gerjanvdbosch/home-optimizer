from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
from optuna import Study, Trial
from skforecast.base import ForecasterBase
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.utils import load_forecaster, save_forecaster

from domain.models.interface import Forecaster
from domain.models.state import BacktestResult


class SkforecastForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()

    @abstractmethod
    def create(self) -> ForecasterBase: ...

    @property
    def backtest_function(self) -> Callable[..., Any]:
        return backtesting_forecaster

    @property
    def tune_function(self) -> Callable[..., Any]:
        return bayesian_search_forecaster

    @abstractmethod
    def arguments(self, df: pd.DataFrame) -> dict[str, Any]: ...

    @abstractmethod
    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def search_space(self, trial: Trial) -> dict[str, Any]: ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().set_index("time").sort_index().asfreq("15min")

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        self.forecaster.fit(**self.arguments(df))

    def predict(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
        steps: int = 24,
    ) -> pd.Series:
        if df is None:
            df = pd.DataFrame()

        df = self.prepare(df)

        return self.forecaster.predict(
            steps=steps,
            last_window=last_window,
            **self.predict_arguments(df),
        )

    def backtest(
        self,
        df: pd.DataFrame,
        steps: int = 24,
    ) -> BacktestResult:
        df = self.prepare(df)

        metric, result = self.backtest_function(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            **self.arguments(df),
        )

        return self.backtest_result(
            df=df,
            metric=metric,
            result=result,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df)

        return self.tune_function(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            search_space=self.search_space,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
            kwargs_create_study={
                "study_name": self.name,
                "storage": study_storage,
                "load_if_exists": True,
                "direction": "minimize",
            },
            **self.arguments(df),
        )

    def create_cv(self, df: pd.DataFrame, steps: int) -> TimeSeriesFold:
        return TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=True,
            fixed_train_size=False,
        )

    def backtest_result(
        self,
        df: pd.DataFrame,
        metric: pd.DataFrame,
        result: pd.DataFrame,
    ) -> BacktestResult:
        result = result.copy()

        result["actual"] = df.loc[result.index, self.target_column]

        return BacktestResult(
            name=self.name,
            y_axis=self.y_axis,
            unit=self.unit,
            mae=float(metric["mean_absolute_error"].iloc[0]),
            points=[
                {
                    "time": str(index),
                    **{str(key): float(value) for key, value in row.items()},
                }
                for index, row in result.iterrows()
            ],
        )

    def save(self, path: Path) -> None:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_forecaster(
            self.forecaster,
            str(path / f"{self.name}.joblib"),
        )

    def load(self, path: Path) -> None:
        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = cast(
            ForecasterBase,
            load_forecaster(str(file_name)),
        )
