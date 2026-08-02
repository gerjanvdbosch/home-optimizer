from dataclasses import dataclass
from typing import Any

import pandas as pd
from optuna import Trial
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from features.dataset import DatasetBuilder
from features.forecaster import SklearnForecaster


@dataclass
class SolarFeatureGenerator:
    n_revisions: int = 4
    epsilon: float = 1e-6

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_columns = {
            "time",
            "target_time",
            "p10",
            "p50",
            "p90",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if self.n_revisions < 1:
            raise ValueError("n_revisions must be >= 1")

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

        df = df.sort_values(["target_time", "time"]).reset_index(drop=True)

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600

        df["spread"] = df["p90"] - df["p10"]
        df["spread_relative"] = df["spread"] / (df["p50"] + self.epsilon)

        group = df.groupby("target_time", sort=False)

        for n in range(1, self.n_revisions + 1):
            suffix = f"_previous_{n}"

            df[f"p10{suffix}"] = group["p10"].shift(n)
            df[f"p50{suffix}"] = group["p50"].shift(n)
            df[f"p90{suffix}"] = group["p90"].shift(n)

            previous_time = group["time"].shift(n)

            df[f"time{suffix}"] = previous_time

            df[f"age_hours{suffix}"] = (
                df["time"] - previous_time
            ).dt.total_seconds() / 3600

            df[f"revision_change_{n}"] = df["p50"] - df[f"p50{suffix}"]

            previous = df[f"p50{suffix}"]

            df[f"revision_change_relative_{n}"] = df[
                f"revision_change_{n}"
            ] / previous.where(previous.abs() > self.epsilon)

            df[f"p10_delta{suffix}"] = df["p10"] - df[f"p10{suffix}"]
            df[f"p90_delta{suffix}"] = df["p90"] - df[f"p90{suffix}"]

            df[f"spread_delta{suffix}"] = df["spread"] - (
                df[f"p90{suffix}"] - df[f"p10{suffix}"]
            )

        print(
            df[
                [
                    "time",
                    "target_time",
                    "p50",
                    "p50_previous_1",
                    "p50_previous_2",
                    "p50_previous_3",
                    "age_hours_previous_1",
                    "age_hours_previous_2",
                    "age_hours_previous_3",
                    "revision_change_1",
                    "revision_change_2",
                    "revision_change_3",
                    "revision_change_relative_1",
                ]
            ].head(20)
        )

        print(
            df[df["target_time"] == "2026-07-21 12:00:00+00:00"][
                [
                    "time",
                    "target_time",
                    "p50",
                    "p50_previous_1",
                    "p50_previous_2",
                    "p50_previous_3",
                    "age_hours_previous_1",
                    "age_hours_previous_2",
                    "age_hours_previous_3",
                    "revision_change_1",
                    "revision_change_2",
                    "revision_change_3",
                    "revision_change_relative_1",
                ]
            ]
        )

        return df


class SolarForecaster(SklearnForecaster):
    feature_generator = SolarFeatureGenerator()

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
        return "error"

    @property
    def evaluation_column(self) -> str:
        return "P_solar"

    @property
    def exog_columns(self) -> list[str]:
        revision_features = []

        for n in range(1, self.feature_generator.n_revisions + 1):
            revision_features.extend(
                [
                    f"revision_change_{n}",
                    f"revision_change_relative_{n}",
                    f"age_hours_previous_{n}",
                ]
            )

        return [
            "p10",
            "p50",
            "p90",
            "spread",
            "spread_relative",
            "lead_time_hours",
            *revision_features,
        ]

    def create(self):
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=150,
            min_samples_leaf=15,
            random_state=42,
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df[df["target_time"] >= df["time"]]

        df = self.feature_generator.transform(df)

        df["error"] = df["P_solar"] - df["p50"]
        df["error_relative"] = df["error"] / (
            df["p50"] + self.feature_generator.epsilon
        )

        df = df.dropna(
            subset=[
                "P_solar",
                "p50",
                "error",
            ]
        )

        print(
            df[df["target_time"] == "2026-07-21 12:00:00+00:00"][
                [
                    "time",
                    "target_time",
                    "lead_time_hours",
                    "p50",
                    "p10",
                    "p90",
                    "P_solar",
                    "error",
                    "error_relative",
                ]
            ]
        )

        return df

    def predict_result(
        self,
        prediction: pd.Series,
        df: pd.DataFrame,
    ) -> pd.Series:
        result = df["p50"] + prediction
        result.name = "P_solar_corrected"

        baseline_mae = (result["P_solar"] - result["p50"]).abs().mean()
        model_mae = (result["P_solar"] - result["pred"]).abs().mean()

        print(baseline_mae)
        print(model_mae)

        return result

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
                "P_solar",
                config.solar.production,
                interval="15m",
                aggregation="mean",
            )
            .attribute_timeseries(
                "p10",
                config.solar.forecast.p10,
                interval="15m",
                aggregation="last",
                target_interval="15min",
            )
            .attribute_timeseries(
                "p50",
                config.solar.forecast.p50,
                interval="15m",
                aggregation="last",
                target_interval="15min",
            )
            .attribute_timeseries(
                "p90",
                config.solar.forecast.p90,
                interval="15m",
                aggregation="last",
                target_interval="15min",
            )
            .join("p50", "p10", on=("time", "target_time"), how="inner")
            .join("p50", "p90", on=("time", "target_time"), how="inner")
            .join(
                "p50",
                "P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .build()
        )
