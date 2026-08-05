import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
        return "error"

    @property
    def exog_columns(self) -> list[str]:
        return [
            "p10",
            "p50",
            "p90",
        ]

    def create(self):
        return HistGradientBoostingRegressor()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["target_time"] > df["time"]].copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600

        df["spread"] = df["p90"] - df["p10"]
        df["spread_relative"] = df["spread"] / (df["p50"])

        df["error"] = df["P_solar"] - df["p50"]
        df["error_relative"] = df["error"] / df["p50"]

        print(
            df[
                [
                    "time",
                    "target_time",
                    "lead_time_hours",
                    "P_solar",
                    "p10",
                    "p50",
                    "p90",
                    "spread",
                    "spread_relative",
                    "error",
                    "error_relative",
                ]
            ]
        )

        df = df.sort_values(["time", "target_time"])

        return df

    def predict_result(
        self,
        prediction: pd.Series,
        df: pd.DataFrame,
    ) -> pd.Series:
        return prediction

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

        backtest_points = [
            BacktestPoint(
                label="Actual",
                points=make_points(
                    df.drop_duplicates("target_time"),
                    "P_solar",
                ),
            )
        ]

        for update_time, update_df in df.sort_values("time").groupby("time"):
            ts = pd.to_datetime(str(update_time))
            label = to_local_time(ts.to_pydatetime()).strftime("%H:%M")
            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label}",
                    points=make_points(
                        update_df.sort_values("target_time"),
                        "p50",
                    ),
                )
            )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=0,
            points=backtest_points,
        )

    def backtest_result(self, result: pd.DataFrame) -> BacktestResult:
        raise NotImplementedError()

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_solar",
                config.solar.production,
                interval="30m",
                aggregation="mean",
            )
            # .timeseries(
            #     "P_solar_min",
            #     config.solar.production,
            #     interval="30min",
            #     aggregation="min",
            # )
            # .timeseries(
            #     "P_solar_max",
            #     config.solar.production,
            #     interval="30min",
            #     aggregation="max",
            # )
            .attribute_timeseries(
                "p10",
                config.solar.forecast.p10,
                interval="15m",
                aggregation="last",
                target_interval="30min",
            )
            .attribute_timeseries(
                "p50",
                config.solar.forecast.p50,
                interval="15m",
                aggregation="last",
                target_interval="30min",
            )
            .attribute_timeseries(
                "p90",
                config.solar.forecast.p90,
                interval="15m",
                aggregation="last",
                target_interval="30min",
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
