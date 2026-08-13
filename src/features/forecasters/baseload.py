from typing import Any

import numpy as np
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer

from domain.models import Config, DatasetDefinition, ForecasterType, SensorReference
from features.dataset import DatasetBuilder
from features.forecaster import SkforecastForecaster


class BaseloadForecaster(SkforecastForecaster):
    @property
    def name(self) -> ForecasterType:
        return "baseload"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "P_baseload"

    @property
    def exog_columns(self) -> list[str]:
        return []

    def create(self, **overrides: Any):
        return ForecasterRecursive(
            forecaster_id=overrides.pop("forecaster_id", self.name),
            estimator=overrides.pop(
                "estimator",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.03,
                    max_depth=8,
                    max_iter=100,
                    min_samples_leaf=5,
                    l2_regularization=10.0,
                    random_state=42,
                ),
            ),
            lags=overrides.pop("lags", [1, 2, 3, 4, 95, 96, 97, 671, 672, 673]),
            calendar_features=overrides.pop(
                "calendar_features",
                CalendarFeatures(
                    features=["hour", "day_of_week", "weekend"],
                    encoding="cyclical",
                ),
            ),
            window_features=overrides.pop(
                "window_features",
                RollingFeatures(
                    stats=["mean", "mean", "std", "std", "min", "min"],
                    window_sizes=[16, 96, 16, 96, 16, 96],
                ),
            ),
            # transformer_y=FunctionTransformer(func=np.sqrt, inverse_func=np.square),
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
            **overrides,
        )

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.025, 0.06, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 5, 8),
            "max_iter": trial.suggest_int(
                "max_iter",
                20,
                180,
                step=40,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                5,
                20,
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1.0, 20.0, log=True
            ),
        }

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_baseload",
                SensorReference(entity_id="sensor.stroomverbruik_base_load"),
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .build()
        )
