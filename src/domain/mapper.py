from datetime import datetime, timezone

import pandas as pd

from domain.models.state import (
    BoilerMeasurement,
    Forecast,
    HeatPumpMeasurement,
    Measurements,
    OptimizerState,
    SeriesPoint,
    SolarMeasurement,
    SolcastForecast,
)


class StateMapper:
    def map(self, df: pd.DataFrame) -> OptimizerState:
        return OptimizerState(
            updated=datetime.now(timezone.utc),
            measurements=Measurements(
                solar=SolarMeasurement(
                    production=self._parse_series(df, "pv_production")
                ),
                heat_pump=HeatPumpMeasurement(
                    state=self._parse_series(df, "heat_pump_state"),
                    boiler=BoilerMeasurement(
                        top_temperature=self._parse_series(
                            df, "boiler_top_temperature"
                        ),
                        bottom_temperature=self._parse_series(
                            df, "boiler_bottom_temperature"
                        ),
                    ),
                ),
            ),
            forecast=Forecast(
                solcast=SolcastForecast(
                    p10=self._parse_series(df, "solar_p10"),
                    p50=self._parse_series(df, "solar_p50"),
                    p90=self._parse_series(df, "solar_p90"),
                )
            ),
        )

    def _parse_series(self, df: pd.DataFrame, column: str) -> list[SeriesPoint]:
        if column not in df.columns:
            return []

        return [
            SeriesPoint(
                time=row["time"],
                value=row[column],
            )
            for _, row in df[["time", column]].dropna().iterrows()
        ]
