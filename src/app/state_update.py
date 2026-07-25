from datetime import datetime, timezone

from domain.models.config import HomeConfig
from domain.models.interface import Storage
from domain.models.state import OptimizerState
from domain.parser import parse_timeseries
from features.dataset.builder import DatasetLoader


class StateUpdater:
    def __init__(self, loader: DatasetLoader, storage: Storage):
        self.loader = loader
        self.storage = storage

    def load(self) -> OptimizerState:
        return OptimizerState(**self.storage.load())

    def update(self, config: HomeConfig) -> None:
        now = datetime.now(timezone.utc)

        # state = OptimizerState(
        #     updated=now,
        #     solar_forecast=,
        #     pv_production=parse_timeseries(production_points),
        # )

        # self.storage.save(state.model_dump())
