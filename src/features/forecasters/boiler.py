from typing import Any

import pandas as pd
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive
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
    def target_column(self) -> str:
        return "T_top"

    @property
    def exog_columns(self) -> list[str]:
        return [
            "T_bottom",
            "state",
            "compressor_freq",
            "T_setpoint",
            "T_supply",
        ]

    def create(self):
        return ForecasterRecursive(
            forecaster_id=self.name,
            estimator=HistGradientBoostingRegressor(
                learning_rate=0.031,
                max_depth=5,
                max_iter=250,
                min_samples_leaf=16,
                random_state=42,
            ),
            lags=48,
            # window_features=RollingFeatures(
            #     stats=[
            #         "mean",
            #         "mean",
            #         "mean",
            #         "mean",
            #         "mean",
            #     ],
            #     window_sizes=[
            #         4,
            #         16,
            #         48,
            #         96,
            #         192,
            #     ],
            # ),
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
            "y": df[self.target_column],
            "exog": df[self.exog_columns],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {"last_window": last_window[self.exog_columns]}

    def backtest_arguments(self, df: pd.DataFrame):
        return {
            "y": df[self.target_column],
            "exog": df[self.exog_columns],
        }

    def backtest_results(
        self,
        df: pd.DataFrame,
        metric: pd.DataFrame,
        result: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        result = result.copy()

        result["actual"] = df.loc[
            result.index,
            self.target_column,
        ]

        return metric, result

    def tune_arguments(self, df: pd.DataFrame):
        return {
            "y": df[self.target_column],
            "exog": df[self.exog_columns],
        }

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "lags": trial.suggest_categorical(
                "lags",
                [
                    48,
                    96,
                    192,
                ],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.03,
                0.07,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                2,
                5,
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                50,
                250,
                step=50,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                10,
                40,
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
