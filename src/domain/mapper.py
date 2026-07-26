from datetime import datetime, timezone

import pandas as pd

from domain.models.state import (
    BoilerMeasurements,
    ForecastState,
    HeatPumpMeasurements,
    MeasurementState,
    OptimizerState,
    SeriesPoint,
    SolarForecast,
    SolarMeasurements,
)


class StateMapper:
    def map(self, df: pd.DataFrame) -> OptimizerState:
        return OptimizerState(
            updated=datetime.now(timezone.utc),
            measurements=MeasurementState(
                solar=SolarMeasurements(production=self.parse_series(df, "pv_production")),
                heat_pump=HeatPumpMeasurements(
                    boiler=BoilerMeasurements(
                        top_temperature=self.parse_series(df, "boiler_top_temperature"),
                        bottom_temperature=self.parse_series(df, "boiler_bottom_temperature"),
                    ),
                ),
            ),
            forecast=ForecastState(
                solar=SolarForecast(
                    p10=self.parse_series(df, "solar_p10"),
                    p50=self.parse_series(df, "solar_p50"),
                    p90=self.parse_series(df, "solar_p90"),
                )
            ),
        )

    def parse_series(self, df: pd.DataFrame, column: str) -> list[SeriesPoint]:
        if column not in df.columns:
            return []

        return [
            SeriesPoint(
                time=row["time"],
                value=row[column],
            )
            for _, row in df[["time", column]].dropna().iterrows()
        ]
