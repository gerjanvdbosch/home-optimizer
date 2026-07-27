from typing import Any

import pandas as pd
from optuna import Trial
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import AppConfig, ForecasterType
from domain.models.dataset import DatasetDefinition
from features.dataset import DatasetBuilder
from features.forecaster import SeriesForecaster


class BoilerForecaster(SeriesForecaster):
    @property
    def name(self) -> ForecasterType:
        return "boiler"

    def create(self):
        return ForecasterDirectMultiVariate(
            estimator=HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
            ),
            level="T_top",
            steps=96,
            lags=24,
            window_features=RollingFeatures(
                stats=["mean", "std", "min", "max"],
                window_sizes=[4, 16, 48, 96],
            ),
            calendar_features=CalendarFeatures(
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
            ),
        )

    def fit_arguments(self, df: pd.DataFrame):
        return {
            "series": df[["T_top", "T_bottom"]],
            "exog": df[["state"]],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {"last_window": last_window[["T_top", "T_bottom"]]}

    def backtest_arguments(self, df: pd.DataFrame):
        return {
            "series": df[["T_top", "T_bottom"]],
            "exog": df[["state"]],
        }

    def tune_arguments(self, df: pd.DataFrame):
        return {
            "series": df[["T_top", "T_bottom"]],
            "exog": df[["state"]],
        }

    def search_space(self, trial: Trial) -> dict[str, Any]:
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

    def dataset(self, config: AppConfig) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            .build()
        )
