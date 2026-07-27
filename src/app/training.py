from datetime import datetime, timedelta, timezone
from pathlib import Path

from domain.models.config import AppConfig, BacktestRequest, ForecasterType, TrainRequest
from domain.models.interface import Forecaster
from features.dataset import DatasetLoader


class Trainer:
    def __init__(
        self,
        loader: DatasetLoader,
        path: Path,
        forecasters: list[Forecaster],
    ):
        self.loader = loader
        self.path = path
        self.forecasters = forecasters

    def train(self, config: AppConfig, request: TrainRequest):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        for forecaster in self.forecasters:
            if request.forecaster and forecaster.name != request.forecaster:
                continue

            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            forecaster.fit(df)
            forecaster.save(self.path)

    def backtest(self, config: AppConfig, request: BacktestRequest):
        forecaster = self._get_forecaster(request.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        result, result2 = forecaster.backtest(df)

        print(result)
        print(result2)

    def tune(self, request: TrainRequest):
        pass

    def _get_forecaster(self, name: ForecasterType) -> Forecaster:
        for forecaster in self.forecasters:
            if forecaster.name == name:
                forecaster.load(self.path)
                return forecaster

        raise ValueError(f"Unknown forecaster: {name}")
