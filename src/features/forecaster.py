from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from optuna import Study
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor


class SolarForecaster:
    def __init__(self):
        # self.forecaster =
        pass

    def fit(self):
        pass

    def backtest(self):
        pass

    def predict(self):
        pass

    def save(self, path: str | Path):
        joblib.dump(
            self,
            path,
        )

    def load(self, path: str | Path):
        loaded = joblib.load(path)


class DhwForecaster:

    def __init__(self, lags: int = 48, model=None):
        if model is None:
            model = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
            )

        window_features = RollingFeatures(
            stats=[
                "mean",
                "std",
                "min",
                "max",
            ],
            window_sizes=[
                4,
                16,
                48,
                96,
            ],
        )

        calendar_features = CalendarFeatures(
            features=[
                "minute",
                "hour",
                "week",
                "month",
                "quarter",
                "day_of_week",
                "weekend",
            ],
            encoding="cyclical",
        )

        self.forecaster = ForecasterRecursive(
            estimator=model,
            lags=lags,
            calendar_features=calendar_features,
            window_features=window_features,
        )

        self.best_params = None
        self.best_score = None
        self.tuning_results = None

    def fit(self, temperature: pd.Series, exog: pd.DataFrame | None = None):
        temperature = temperature.sort_index()

        if exog is not None:
            exog = exog.sort_index()

        self.forecaster.fit(
            y=temperature,
            exog=exog,
        )

        return self

    def backtest(
        self,
        temperature: pd.Series,
        exog: pd.DataFrame | None = None,
        steps: int = 24,
    ):

        temperature = temperature.sort_index()

        if exog is not None:
            exog = exog.sort_index()

        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(temperature) * 0.7),
            refit=False,
            fixed_train_size=False,
        )

        metrics, predictions = backtesting_forecaster(
            forecaster=self.forecaster,
            y=temperature,
            exog=exog,
            cv=cv,
            metric="mean_absolute_error",
        )

        return metrics, predictions

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame,
    ):
        return self.forecaster.predict(
            steps=steps,
            exog=exog,
        )

    def save(self, path: str | Path):
        joblib.dump(
            self,
            path,
        )

    def load(self, path: str | Path):
        return joblib.load(path)

    def tune(
        self,
        temperature: pd.Series,
        exog: pd.DataFrame | None = None,
        steps: int = 24,
        n_trials: int = 20,
    ):
        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(temperature) * 0.7),
            refit=False,
        )

        def search_space(trial):
            return {
                "lags": trial.suggest_categorical("lags", [24, 48, 72, 96]),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    10,
                ),
                "max_iter": trial.suggest_int(
                    "max_iter",
                    100,
                    500,
                    step=50,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    10,
                    80,
                ),
            }

        results, study = bayesian_search_forecaster(
            forecaster=self.forecaster,
            y=temperature,
            exog=exog,
            cv=cv,
            metric="mean_absolute_error",
            search_space=search_space,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
        )

        study = cast(Study, study)

        best_trial = study.best_trial

        self.tuning_results = results
        self.best_params = best_trial.params
        self.best_score = best_trial.value

        return self
