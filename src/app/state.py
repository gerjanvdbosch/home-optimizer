from datetime import datetime, timezone

from domain.models.config import AppConfig
from domain.models.state import OptimizerState
from features.dataset import DatasetLoader
from infrastructure.repository import StateRepository


class StateManager:
    def __init__(self, loader: DatasetLoader, repository: StateRepository):
        self.loader = loader
        self.repository = repository

    def load(self) -> OptimizerState:
        return self.repository.load()

    def update(self, config: AppConfig) -> None:
        now = datetime.now(timezone.utc)

        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        # dataset = self._dataset()

        # data = self.loader.load(dataset, start, now)

        # data = mapper.transform(data)
        # print(data)

        # self.repository.save(state)
