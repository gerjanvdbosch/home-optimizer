from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from domain.models import (
    Config,
    DatasetDefinition,
    SeriesPoint,
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

    def update_schedule(
        self,
        schedule: Sequence[int],
        power_kw: float,
        times: list[datetime],
    ) -> None:
        boiler_power = [
            SeriesPoint(time=t, value=float(val) * power_kw)
            for t, val in zip(times, schedule, strict=False)
        ]

        state = self.load()
        state.schedule.heat_pump.power = boiler_power

        self.repository.save(state)

    def _map(
        self,
        df: pd.DataFrame,
        existing: State | None = None,
    ) -> State:
        state = State(updated=datetime.now(timezone.utc))

        if existing is not None:
            state.predictions = existing.predictions
            state.schedule = existing.schedule

        state.measurements.solar.production = self._parse_series(df, "pv_production")
        state.measurements.heat_pump.state = self._parse_series(df, "heat_pump_state")
        state.measurements.heat_pump.boiler.top_temperature = self._parse_series(
            df, "boiler_top_temperature"
        )
        state.measurements.heat_pump.boiler.bottom_temperature = self._parse_series(
            df, "boiler_bottom_temperature"
        )

        for forecast_source in ["solcast", "open_meteo"]:
            source_obj = getattr(state.forecast, forecast_source)
            for attr, _ in source_obj.items():
                setattr(source_obj, attr, self._parse_series(df, attr))

        return state

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
                attributes=["temperature"],
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
