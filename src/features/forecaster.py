from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from optuna import Trial
from skforecast.base import ForecasterBase
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)

from domain.models.interface import Forecaster


class BaseForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()

    @abstractmethod
    def create(self) -> ForecasterBase: ...

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

        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=False,
            fixed_train_size=False,
        )

        return backtesting_forecaster(
            forecaster=self.forecaster,
            cv=cv,
            **self.backtest_arguments(df),
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
    ):
        df = self.prepare(df)

        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=False,
        )

        # study = cast(Study, study)

        # best_trial = study.best_trial
        #
        # self.tuning_results = results
        # self.best_params = best_trial.params
        # self.best_score = best_trial.value

        # return results, study

        return bayesian_search_forecaster(
            forecaster=self.forecaster,
            cv=cv,
            search_space=self.search_space,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
            **self.tune_arguments(df),
        )

    def save(self, path: Path) -> None:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.forecaster,
            path / f"{self.name}.joblib",
        )

    def load(self, path: Path) -> None:
        self.forecaster = joblib.load(path)
