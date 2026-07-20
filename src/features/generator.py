from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SolarForecastFeatureGenerator:
    forecast_columns: list[str] = field(default_factory=lambda: ["p10", "p50", "p90"])
    forecast_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])

    include_spread: bool = True
    include_lags: bool = True
    include_delta: bool = True
    include_lead_time: bool = True

    issue_column: str = "issue_time"
    target_column: str = "target_time"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[self.issue_column] = pd.to_datetime(df[self.issue_column])
        df[self.target_column] = pd.to_datetime(df[self.target_column])

        df = df.sort_values([self.target_column, self.issue_column])

        if self.include_spread:
            df["spread"] = df["p90"] - df["p10"]
            df["spread_relative"] = (df["p90"] - df["p10"]) / df["p50"].clip(lower=1)

        if self.include_lead_time:
            df["lead_time"] = (
                df[self.target_column] - df[self.issue_column]
            ).dt.total_seconds() / 3600.0

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

                    if self.include_spread:
                        df[f"spread_delta_{lag}"] = df["spread"] - df[f"spread_lag_{lag}"]

        return df
