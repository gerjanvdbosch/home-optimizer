import logging

from pydantic import ValidationError

from domain.models import BacktestResult, Config, State
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

    def save(self, state: State) -> None:
        self.storage.save(state)

    def load(self) -> State:
        data = self.storage.load()

        if not data:
            return State()

        return State.model_validate(data)


class BacktestRepository:
    def __init__(self, storage: JsonStorage):
        self.storage = storage

    def save(self, df: BacktestResult):
        self.storage.save(df)

    def load(self) -> BacktestResult | None:
        data = self.storage.load()

        if not data:
            return

        try:
            return BacktestResult.model_validate(data)

        except ValidationError as e:
            logging.warning("Invalid backtest result: %s", e)

            return None

    def clear(self) -> None:
        self.storage.remove()
