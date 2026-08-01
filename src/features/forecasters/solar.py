from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from features.dataset import DatasetBuilder
from features.forecaster import BaseForecaster


class SolarForecaster(BaseForecaster):
    @property
    def name(self) -> ForecasterType:
        return "solar"

    @property
    def y_axis(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "production"

    @property
    def exog_columns(self) -> list[str]:
        return [
            "p10",
            "p50",
            "p90",
            "spread",
            "spread_relative",
            "lead_time",
            "p50_asof_2h",
            "p50_asof_4h",
            "p50_asof_8h",
            "p50_asof_12h",
            "p50_asof_24h",
            "p50_delta_2h",
            "p50_delta_4h",
            "p50_delta_8h",
            "p50_delta_12h",
            "p50_delta_24h",
            "p50_delta_relative_2h",
            "p50_delta_relative_4h",
            "p50_delta_relative_8h",
            "p50_delta_relative_12h",
            "p50_delta_relative_24h",
        ]

    def create(self):
        return ForecasterRecursive(
            forecaster_id=self.name,
            estimator=HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=5,
                max_iter=150,
                min_samples_leaf=15,
                random_state=42,
            ),
            lags=48,
            calendar_features=CalendarFeatures(
                features=[
                    "minute",
                    "hour",
                    "month",
                    "day_of_week",
                    "weekend",
                ],
                encoding="cyclical",
            ),
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = df[df["target_time"] >= df["time"]]

        df = df.dropna(subset=[self.target_column])

        print(df)

        # forecast_columns = ["p10", "p50", "p90"]
        #
        # df["time"] = pd.to_datetime(df["time"])
        # df["target_time"] = pd.to_datetime(df["target_time"])
        #
        # result = []
        #
        # for forecast_time, group in df.groupby("time"):
        #     group = group.set_index("target_time").sort_index()
        #
        #     group = group[forecast_columns].resample("15min").ffill()
        #
        #     group["time"] = forecast_time
        #
        #     result.append(group.reset_index())
        #
        # return pd.concat(result, ignore_index=True)

        return df

    def arguments(self, df: pd.DataFrame):
        return {
            "y": df[self.target_column],
            "exog": df[self.exog_columns],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {
            "last_window": last_window,
            "exog": df[self.exog_columns] if df is not None else None,
        }

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "lags": trial.suggest_categorical(
                "lags",
                [
                    16,
                    32,
                    48,
                    96,
                ],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.03,
                0.08,
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
                "production",
                config.solar.production,
                interval="15m",
                aggregation="mean",
                fill="none",
            )
            .attribute_timeseries(
                "p10",
                config.solar.forecast.p10,
                interval="1m",
                aggregation="last",
                target_interval="15min",
            )
            .attribute_timeseries(
                "p50",
                config.solar.forecast.p50,
                interval="1m",
                aggregation="last",
                target_interval="15min",
            )
            .attribute_timeseries(
                "p90",
                config.solar.forecast.p90,
                interval="1m",
                aggregation="last",
                target_interval="15min",
            )
            .join("p50", "p10", on=("time", "target_time"), how="inner")
            .join("p50", "p90", on=("time", "target_time"), how="inner")
            .join(
                "p50",
                "production",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .build()
        )


@dataclass
class SolarFeatureGenerator:
    forecast_columns: list[str] = field(default_factory=lambda: ["p10", "p50", "p90"])

    forecast_ages: list[str] = field(
        default_factory=lambda: [
            "2h",
            "4h",
            "8h",
            "12h",
            "24h",
        ]
    )

    epsilon: float = 10.0

    include_spread: bool = True
    include_lags: bool = True
    include_delta: bool = True
    include_time: bool = True

    time_column: str = "time"
    target_column: str = "target_time"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df[self.target_column] = pd.to_datetime(df[self.target_column])

        df = df.sort_values([self.target_column, self.time_column]).reset_index(
            drop=True
        )

        if self.include_spread:
            df["spread"] = df["p90"] - df["p10"]
            df["spread_relative"] = (df["p90"] - df["p10"]) / (df["p50"] + self.epsilon)

        if self.include_time:
            df["lead_time"] = (
                df[self.target_column] - df[self.time_column]
            ).dt.total_seconds() / 60

        lag_columns = list(self.forecast_columns)

        if self.include_spread:
            lag_columns.append("spread")

        if self.include_lags:
            lookup = (
                df[[self.target_column, self.time_column, *lag_columns]]
                .sort_values(self.time_column)
                .reset_index(drop=True)
                .rename(columns={self.time_column: "_matched_time"})
            )

            for age in self.forecast_ages:
                key = age
                delta = pd.to_timedelta(age)

                query = df[[self.target_column, self.time_column]].reset_index()
                query["_query_time"] = pd.to_datetime(query[self.time_column]) - delta
                query = query.sort_values("_query_time")

                merged = (
                    pd.merge_asof(
                        query,
                        lookup,
                        left_on="_query_time",
                        right_on="_matched_time",
                        by=self.target_column,
                        direction="backward",
                    )
                    .set_index("index")
                    .sort_index()
                )

                for col in lag_columns:
                    df[f"{col}_asof_{key}"] = merged[col]

                df[f"age_actual_{key}"] = (
                    df[self.time_column] - merged["_matched_time"]
                ).dt.total_seconds() / 60

        if self.include_delta and self.include_lags:
            for age in self.forecast_ages:
                key = age

                df[f"p50_delta_{key}"] = df["p50"] - df[f"p50_asof_{key}"]
                df[f"p50_delta_relative_{key}"] = df[f"p50_delta_{key}"] / (
                    df[f"p50_asof_{key}"] + self.epsilon
                )

                if self.include_spread:
                    df[f"spread_delta_{key}"] = df["spread"] - df[f"spread_asof_{key}"]

        print(
            df[
                [
                    "target_time",
                    "time",
                    "p50",
                    "p50_asof_2h",
                    "p50_asof_4h",
                    "p50_asof_8h",
                ]
            ].head(20)
        )

        print(
            df[df["target_time"] == "2026-07-21 12:00:00+00:00"][
                [
                    "target_time",
                    "time",
                    "p50",
                    "p50_asof_2h",
                    "p50_asof_4h",
                ]
            ]
        )

        return df
