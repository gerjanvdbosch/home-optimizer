from pathlib import Path

import joblib
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
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
                "min",
                "max",
            ],
            window_sizes=[
                4,
                16,
                96,
            ],
        )

        calendar_features = CalendarFeatures(
            features=[
                "hour",
                "day_of_week",
                "week",
                "weekend",
                "month",
                "quarter",
            ],
        )

        self.forecaster = ForecasterRecursive(
            estimator=model,
            lags=lags,
            calendar_features=calendar_features,
            window_features=window_features,
        )

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
