from datetime import datetime, timezone

import pandas as pd

from domain.models import (
    BoilerMeasurement,
    Config,
    DatasetDefinition,
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
from features.dataset import DatasetBuilder, DatasetLoader
from infrastructure.repository import StateRepository


class StateManager:
    def __init__(
        self,
        loader: DatasetLoader,
        repository: StateRepository,
    ):
        self.loader = loader
        self.repository = repository

    def load(self) -> State:
        return self.repository.load()

    def update(self, config: Config) -> None:
        now = datetime.now(timezone.utc)

        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        df = self.loader.load(
            self._dataset(config),
            start,
            now,
        )

        state = self._map(df, self.load())

        self.repository.save(state)

    def update_prediction(self, name: str, series: pd.Series) -> None:
        state = self.load()

        points = [
            SeriesPoint(time=pd.to_datetime(str(time)), value=float(value))
            for time, value in series.items()
        ]

        if hasattr(state.predictions, name):
            setattr(state.predictions, name, points)
        else:
            raise ValueError(f"Unknown prediction: '{name}'")

        self.repository.save(state)

    def _map(
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

    def _dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .attribute_series(
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
            )
            .attribute_series(
                "open_meteo",
                config.forecast.open_meteo,
                attributes=[
                    "gti",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                ],
                target_resample="mean",
                target_interval="30min",
                target_label="right",
                target_closed="right",
                target_shift=True,
            )
            .timeseries(
                "heat_pump_state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            .timeseries(
                "pv_production",
                config.solar.production,
                aggregation="mean",
                interval="15m",
            )
            .timeseries(
                "boiler_top_temperature",
                config.heat_pump.boiler.top_temperature,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "boiler_bottom_temperature",
                config.heat_pump.boiler.bottom_temperature,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .build()
        )
