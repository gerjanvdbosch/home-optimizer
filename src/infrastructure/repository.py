from domain.models.config import Config
from domain.models.state import BacktestResult, OptimizerState
from infrastructure.storage import JsonStorage


class ConfigRepository:
    def __init__(self, storage: JsonStorage):
        self.storage = storage

    def save(self, config: Config) -> None:
        self.storage.save(config)

    def load(self) -> Config:
        data = self.storage.load()

        if not data:
            raise Exception("No config found, run update first")

        return Config.model_validate(data)


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

    def save(self, df: BacktestResult):
        self.storage.save(df)

    def load(self) -> BacktestResult | None:
        data = self.storage.load()

        if not data:
            return

        return BacktestResult.model_validate(data)
