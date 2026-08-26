from datetime import datetime, timezone

import pandas as pd

from domain.mapper import StateMapper
from domain.models import Config, DatasetDefinition, SeriesPoint, State
from features.dataset import DatasetBuilder, DatasetLoader
from infrastructure.repository import StateRepository


class StateManager:
    def __init__(
        self,
        loader: DatasetLoader,
        repository: StateRepository,
        mapper: StateMapper,
    ):
        self.loader = loader
        self.repository = repository
        self.mapper = mapper

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

        state = self.mapper.map(df, self.load())

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
