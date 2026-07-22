from pathlib import Path

import joblib
import pandas as pd
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SolarForecaster:
    def __init__(
        self,
        lags: int = 4,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:

        calendar = CalendarFeatures(
            features=[
                "month",
                "week",
                "day_of_week",
                "hour",
            ],
            encoding="cyclical",
            keep_original_columns=False,
        )

        self.forecaster = ForecasterRecursive(
            estimator=HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            ),
            lags=lags,
            # calendar_features=calendar,
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

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame,
    ) -> pd.Series:

        return self.forecaster.predict(
            steps=steps,
            exog=exog,
        )

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> dict[str, float]:

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = mean_squared_error(
            y_true,
            y_pred,
        )

        return {
            "mae": mae,
            "rmse": rmse,
        }

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
