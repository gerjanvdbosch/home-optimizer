import pandas as pd

from domain.models.config import AppConfig
from domain.models.state import OptimizerState
from infrastructure.storage import JsonStorage


class ConfigRepository:
    def __init__(self, storage: JsonStorage):
        self.storage = storage

    def save(self, config: AppConfig) -> None:
        self.storage.save(config)

    def load(self) -> AppConfig:
        data = self.storage.load()

        if not data:
            raise Exception("No config found, run update first")

        return AppConfig.model_validate(data)


class StateRepository:
    def __init__(self, storage: JsonStorage):
        self.storage = storage

    def save(self, state: OptimizerState) -> None:
        self.storage.save(state)

    def load(self) -> OptimizerState:
        data = self.storage.load()

        if not data:
            return OptimizerState()

        return OptimizerState.model_validate(data)


class BacktestRepository:
    def __init__(self, storage: JsonStorage):
        self.storage = storage

    def save(self, df: pd.DataFrame):
        self.storage.save(df)

    def load(self):
        return self.storage.load()
