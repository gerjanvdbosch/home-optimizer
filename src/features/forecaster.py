from pathlib import Path

import joblib
import pandas as pd
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from sklearn.ensemble import HistGradientBoostingRegressor


class SolarForecaster:

    def __init__(
        self,
        *,
        steps: int = 48,
        lags: int = 4,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:

        self.steps = steps

        self.forecaster = ForecasterDirect(
            estimator=HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            ),
            lags=lags,
            steps=steps,
        )

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame,
    ) -> None:

        self.forecaster.fit(
            y=y,
            exog=exog,
        )

    def backtest(
        self,
        y: pd.Series,
        exog: pd.DataFrame,
    ):

        cv = TimeSeriesFold(
            steps=self.steps,
            initial_train_size=int(len(y) * 0.8),
            refit=True,
        )

        metrics, predictions = backtesting_forecaster(
            forecaster=self.forecaster,
            y=y,
            exog=exog,
            cv=cv,
            metric=[
                "mean_absolute_error",
                "mean_squared_error",
            ],
        )

        return metrics, predictions

    def predict(
        self,
        exog: pd.DataFrame,
    ) -> pd.Series:

        return self.forecaster.predict(
            steps=self.steps,
            exog=exog,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        joblib.dump(
            self,
            path,
        )

    def load(
        self,
        path: str | Path,
    ) -> None:

        loaded = joblib.load(path)

        self.forecaster = loaded.forecaster
