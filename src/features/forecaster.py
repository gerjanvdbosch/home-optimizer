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


class BaseForecaster(Forecaster):
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
    def fit_arguments(self, df: pd.DataFrame) -> dict[str, Any]: ...

    @abstractmethod
    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def backtest_arguments(self, df: pd.DataFrame) -> dict[str, Any]: ...

    @abstractmethod
    def backtest_results(
        self, df: pd.DataFrame, metric: pd.DataFrame, result: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @abstractmethod
    def tune_arguments(self, df: pd.DataFrame) -> dict[str, Any]: ...

    @abstractmethod
    def search_space(self, trial: Trial) -> dict[str, Any]: ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().set_index("time").sort_index().asfreq("15min")

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        self.forecaster.fit(**self.fit_arguments(df))

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
    ):
        df = self.prepare(df)

        metric, result = self.backtest_function(
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            **self.backtest_arguments(df),
        )

        return self.backtest_results(
            df=df,
            metric=metric,
            result=result,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df)

        # study = optuna.create_study(
        #     study_name=f"{self.name}_forecaster",
        #     storage="sqlite:///data/optuna.db",
        #     load_if_exists=True,
        #     direction="minimize",
        # )

        return self.tune_function(
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            search_space=self.search_space,
            # study_name=study.study_name,
            # storage=study._storage,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
            **self.tune_arguments(df),
        )

    def create_cv(self, df: pd.DataFrame, steps: int) -> TimeSeriesFold:
        return TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=True,
            fixed_train_size=False,
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
        self.forecaster = cast(
            ForecasterBase,
            load_forecaster(
                str(path / f"{self.name}.joblib"),
            ),
        )
