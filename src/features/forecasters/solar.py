import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from domain.models.state import BacktestPoint, BacktestResult
from domain.time import to_local_time
from features.dataset import DatasetBuilder
from features.forecaster import SklearnForecaster


class SolarForecaster(SklearnForecaster):
    @property
    def name(self) -> ForecasterType:
        return "solar"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "P_solar"

    @property
    def exog_columns(self) -> list[str]:
        return [
            # "p10",
            "p50",
            # "p90",
            "lead_time_hours",
            "spread",
            "spread_relative",
            # "hour_sin",
            # "hour_cos",
        ]

    def create(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=10.0,
            random_state=42,
        )

    def predict_arguments(self, df: pd.DataFrame, steps: int = 24):
        now = datetime.now(UTC)

        last_window = df[
            (df["target_time"] <= now) & df["P_solar"].notna()
        ].sort_values("target_time")

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
        )

        print(last_window)
        print(future)

        return {
            "X": future[self.exog_columns].iloc[:steps],
        }

    def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.dropna(subset=[self.target_column, *self.exog_columns]).copy()

        df["error"] = df[self.target_column] - df["p50"]

        return {
            "X": df[self.exog_columns],
            "y": df["error"],
        }

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["target_time"] > df["time"]].copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600

        df["spread"] = df["p90"] - df["p10"]
        df["spread_relative"] = df["spread"] / (df["p50"] + 1e-6)

        df = df.sort_values(["target_time", "time"]).copy()

        hour = df["target_time"].dt.hour + df["target_time"].dt.minute / 60

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        day = df["target_time"].dt.dayofyear

        df["day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)

        # print(
        #     df[
        #         [
        #             "time",
        #             "target_time",
        #             "lead_time_hours",
        #             "P_solar",
        #             "p10",
        #             "p50",
        #             "p90",
        #             "spread",
        #             "spread_relative",
        #         ]
        #     ].to_string()
        # )

        df = df.sort_values(["time", "target_time"])

        return df

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df["p50"].to_numpy() + prediction,
            index=df["target_time"],
            name="pred",
        )

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        df = self.prepare(df)

        def make_points(df: pd.DataFrame, value_col: str) -> list[dict]:
            records = df[["target_time", value_col]].to_dict("records")

            return [
                {
                    "time": pd.to_datetime(record["target_time"]).isoformat(),
                    "value": float(record[value_col]),
                }
                for record in records
            ]

        actual = (
            df[["target_time", "P_solar"]]
            .dropna(subset=["P_solar"])
            .drop_duplicates("target_time")
            .sort_values("target_time")
        )

        backtest_points = [
            BacktestPoint(
                label="Actual",
                group="Actual",
                color="white",
                points=make_points(actual, "P_solar"),
            )
        ]

        baseline_errors: list[float] = []
        ml_errors: list[float] = []

        groups = df.sort_values("time").groupby("time")

        progress = tqdm(
            groups,
            total=groups.ngroups,
            desc="Solar backtest",
        )

        for update_time, update_df in progress:
            forecast = (
                update_df[update_df["target_time"] > update_time]
                .sort_values("target_time")
                .drop_duplicates("target_time", keep="last")
            )

            test = forecast[forecast["P_solar"].notna()].iloc[:steps].copy()

            if test.empty:
                continue

            train = df[
                (df["time"] < update_time)
                & (df["target_time"] < update_time)
                & (df["target_time"] > df["time"])
                & df["P_solar"].notna()
            ].copy()

            train = train.dropna(
                subset=[
                    self.target_column,
                    *self.exog_columns,
                ]
            )

            if train.empty:
                continue

            train["error"] = train["P_solar"] - train["p50"]

            model = self.create()

            model.fit(
                train[self.exog_columns],
                train["error"],
            )

            error_prediction = model.predict(
                test[self.exog_columns],
            )

            test["pred"] = test["p50"] + error_prediction

            baseline_errors.extend((test["P_solar"] - test["p50"]).abs().tolist())

            ml_errors.extend((test["P_solar"] - test["pred"]).abs().tolist())

            ts = pd.to_datetime(str(update_time))
            label = to_local_time(ts.to_pydatetime()).strftime("%d-%m %H:%M")

            backtest_points.append(
                BacktestPoint(
                    label=f"ML {label}",
                    group="ML",
                    points=make_points(test, "pred"),
                )
            )

            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label}",
                    group="Solcast",
                    points=make_points(
                        update_df.sort_values("target_time"),
                        "p50",
                    ),
                )
            )

        baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else 0.0

        ml_mae = float(np.mean(ml_errors)) if ml_errors else 0.0

        improvement = (
            100 * (baseline_mae - ml_mae) / baseline_mae if baseline_mae else 0.0
        )

        logging.info(
            "Solar MAE: baseline=%.2f W, ML=%.2f W, improvement=%.1f%%",
            baseline_mae,
            ml_mae,
            improvement,
        )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=ml_mae,
            points=backtest_points,
        )

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_solar",
                config.solar.production,
                interval="30m",
                aggregation="mean",
                fill=0,
            )
            .attribute_timeseries(
                "p10",
                config.solar.forecast.p10,
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "p50",
                config.solar.forecast.p50,
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "p90",
                config.solar.forecast.p90,
                interval="30m",
                aggregation="last",
            )
            .join("p50", "p10", on=("time", "target_time"), how="outer")
            .join("p50", "p90", on=("time", "target_time"), how="outer")
            .join(
                "p50",
                "P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="outer",
            )
            .build()
        )
