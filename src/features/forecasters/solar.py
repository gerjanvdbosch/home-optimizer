import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from domain.models.state import BacktestResult
from features.dataset import DatasetBuilder
from features.forecaster import SklearnForecaster


class SolarForecaster(SklearnForecaster):
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

        points: list[dict[str, object]] = [
            {
                "time": record["time"].isoformat(),
                "target_time": record["target_time"].isoformat(),
                "value": float(record["value"]),
            }
            for record in df[["time", "target_time", "p50"]]
            .rename(columns={"p50": "value"})
            .to_dict(orient="records")
        ]

        return BacktestResult(
            name=self.name,
            y_axis=self.y_axis,
            unit=self.unit,
            mae=0,
            points=points,
        )

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
                how="left",
            )
            .build()
        )
