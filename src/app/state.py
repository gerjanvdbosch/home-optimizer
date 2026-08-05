from datetime import datetime, timezone

from domain.mapper import StateMapper
from domain.models.config import Config
from domain.models.dataset import DatasetDefinition
from domain.models.state import OptimizerState
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

    def load(self) -> OptimizerState:
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

        state = self.mapper.map(df)

        self.repository.save(state)

    def _dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .attribute_series(
                "solar_p10",
                config.solar.forecast.p10,
            )
            .attribute_series(
                "solar_p50",
                config.solar.forecast.p50,
            )
            .attribute_series(
                "solar_p90",
                config.solar.forecast.p90,
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
