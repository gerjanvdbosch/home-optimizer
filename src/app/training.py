from datetime import datetime, timedelta, timezone
from pathlib import Path

from domain.models.config import AppConfig, TrainRequest
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
            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            forecaster.fit(df)
            forecaster.save(self.path)
