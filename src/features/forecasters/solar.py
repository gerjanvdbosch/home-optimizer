from dataclasses import dataclass, field

import pandas as pd

# class SolarForecaster:
#     def __init__(self):
#         # self.forecaster =
#         pass
# def dataset(self, config):
# return (
#     DatasetBuilder()
#     .timeseries(
#         "pv_production",
#         self.config.solar.production,
#         aggregation="mean",
#         interval="30min",
#     )
#     .forecast(
#         "solar_p10",
#         self.config.solar.forecast.p10,
#     )
#     .forecast(
#         "solar_p50",
#         self.config.solar.forecast.p50,
#     )
#     .forecast(
#         "solar_p90",
#         self.config.solar.forecast.p90,
#     )
#     .timeseries(
#         "heat_pump_mode",
#         self.config.heat_pump.mode,
#         aggregation="last",
#     )
#     .timeseries(
#         "boiler_top",
#         self.config.heat_pump.boiler.top_temperature,
#         aggregation="mean",
#     )
#     .timeseries(
#         "boiler_bottom",
#         self.config.heat_pump.boiler.bottom_temperature,
#         aggregation="mean",
#     )
#     .build()
# )


#
#     def fit(self):
#         pass
#
#     def backtest(self):
#         pass
#
#     def predict(self):
#         pass
#
#     def save(self, path: str | Path):
#         joblib.dump(
#             self,
#             path,
#         )
#
#     def load(self, path: str | Path):
#         loaded = joblib.load(path)
#


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

        df = df.sort_values([self.target_column, self.time_column]).reset_index(drop=True)

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
