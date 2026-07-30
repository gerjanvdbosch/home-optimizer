from typing import Any, Callable

import pandas as pd
from optuna import Trial
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import (
    backtesting_forecaster_multiseries,
    bayesian_search_forecaster_multiseries,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from features.dataset import DatasetBuilder
from features.forecaster import BaseForecaster


class BoilerForecaster(BaseForecaster):
    @property
    def name(self) -> ForecasterType:
        return "boiler"

    @property
    def series_columns(self):
        return ["T_top", "T_bottom"]

    @property
    def exog_columns(self):
        return ["state", "compressor_freq", "T_setpoint", "T_supply"]

    def create(self):
        return ForecasterDirectMultiVariate(
            forecaster_id=self.name,
            estimator=HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.045,
                max_depth=3,
                min_samples_leaf=21,
                random_state=42,
            ),
            level="T_top",
            steps=96,
            lags=48,
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

    @property
    def backtest_function(self) -> Callable[..., Any]:
        return backtesting_forecaster_multiseries

    @property
    def tune_function(self) -> Callable[..., Any]:
        return bayesian_search_forecaster_multiseries

    def fit_arguments(self, df: pd.DataFrame):
        return {
            "series": df[self.series_columns],
            "exog": df[self.exog_columns],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {"last_window": last_window[self.series_columns]}

    def backtest_arguments(self, df: pd.DataFrame):
        return {
            "series": df[self.series_columns],
            "exog": df[self.exog_columns],
        }

    def backtest_results(
        self, df: pd.DataFrame, metric: pd.DataFrame, result: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        actual = df.loc[result.index, "T_top"]

        result = result.copy()
        result["actual"] = actual

        return metric, result

    def tune_arguments(self, df: pd.DataFrame):
        return {
            "series": df[self.series_columns],
            "exog": df[self.exog_columns],
        }

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "lags": trial.suggest_categorical(
                "lags",
                [24, 48, 96],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.02,
                0.08,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                6,
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                100,
                300,
                step=50,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                20,
                50,
            ),
        }

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            .timeseries(
                "compressor_freq",
                config.heat_pump.compressor_frequency,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "T_setpoint",
                config.heat_pump.boiler.setpoint,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
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
                "T_supply",
                config.heat_pump.supply_temperature,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .build()
        )
