from datetime import datetime, timezone

import pandas as pd

from domain.models import (
    BoilerMeasurement,
    Forecast,
    HeatPumpMeasurement,
    Measurements,
    OpenMeteoForecast,
    Predictions,
    SeriesPoint,
    SolarMeasurement,
    SolcastForecast,
    State,
)


class StateMapper:
    def map(
        self,
        df: pd.DataFrame,
        existing: State | None = None,
    ) -> State:
        return State(
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
                    p10=self._parse_series(df, "p10"),
                    p50=self._parse_series(df, "p50"),
                    p90=self._parse_series(df, "p90"),
                ),
                open_meteo=OpenMeteoForecast(
                    temperature=self._parse_series(df, "temperature"),
                    gti=self._parse_series(df, "gti"),
                    cloud_cover_low=self._parse_series(df, "cloud_cover_low"),
                    cloud_cover_mid=self._parse_series(df, "cloud_cover_mid"),
                    cloud_cover_high=self._parse_series(df, "cloud_cover_high"),
                    wind_direction=self._parse_series(df, "wind_direction"),
                    wind_speed=self._parse_series(df, "wind_speed"),
                    precipitation=self._parse_series(df, "precipitation"),
                ),
            ),
            predictions=existing.predictions if existing else Predictions(),
            schedule=existing.schedule if existing else None,
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
