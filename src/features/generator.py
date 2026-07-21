from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SolarForecastFeatureGenerator:
    forecast_columns: list[str] = field(default_factory=lambda: ["p10", "p50", "p90"])
    forecast_lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 12])

    epsilon: float = 10.0

    include_spread: bool = True
    include_lags: bool = True
    include_delta: bool = True
    include_lead_time: bool = True

    time_column: str = "time"
    target_column: str = "target_time"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df[self.target_column] = pd.to_datetime(df[self.target_column])

        df = df.sort_values([self.target_column, self.time_column])

        if self.include_spread:
            df["spread"] = df["p90"] - df["p10"]
            df["spread_relative"] = (df["p90"] - df["p10"]) / (df["p50"] + self.epsilon)

        if self.include_lead_time:
            df["lead_time"] = (
                df[self.target_column] - df[self.time_column]
            ).dt.total_seconds() / 60

            df["previous_update"] = (
                df[self.time_column] - df.groupby(self.target_column)[self.time_column].shift(1)
            ).dt.total_seconds() / 60

        lag_columns = list(self.forecast_columns)

        if self.include_spread:
            lag_columns.append("spread")

        grouped = df.groupby(self.target_column, group_keys=False)

        if self.include_lags:
            for col in lag_columns:
                for lag in self.forecast_lags:
                    df[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

        if self.include_delta:
            for lag in self.forecast_lags:
                if self.include_lags:
                    df[f"p50_delta_{lag}"] = df["p50"] - df[f"p50_lag_{lag}"]
                    df[f"p50_delta_relative_{lag}"] = df[f"p50_delta_{lag}"] / (
                        df[f"p50_lag_{lag}"] + self.epsilon
                    )

                    if self.include_spread:
                        df[f"spread_delta_{lag}"] = df["spread"] - df[f"spread_lag_{lag}"]

        print(
            df[
                [
                    "target_time",
                    "time",
                    "p50",
                    "p50_lag_1",
                    "p50_lag_4",
                    "p50_lag_8",
                ]
            ].head(20)
        )

        print(
            df[df["target_time"] == "2026-07-21 12:00:00+00:00"][
                [
                    "target_time",
                    "time",
                    "p50",
                    "p50_lag_1",
                    "p50_lag_2",
                ]
            ]
        )

        return df
