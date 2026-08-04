from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from domain.models.state import BacktestResult
from features.dataset import DatasetBuilder
from features.forecaster import SklearnForecaster


@dataclass
class SolarFeatureGenerator:
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

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600

        df["spread"] = df["p90"] - df["p10"]
        df["spread_relative"] = df["spread"] / (df["p50"])

        print(
            df[df["target_time"] == "2026-07-21 12:00:00+00:00"][
                [
                    "time",
                    "target_time",
                    "p50",
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
        return [
            "p10",
            "p50",
            "p90",
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

        df = df[df["target_time"] > df["time"]]

        df = self.feature_generator.transform(df)

        df["error"] = df["P_solar"] - df["p50"]
        df["error_relative"] = df["error"] / df["p50"]

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
        print(prediction)
        print(df)

        return prediction

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        raise NotImplementedError()

    def backtest_result(self, result: pd.DataFrame) -> BacktestResult:
        raise NotImplementedError()

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
