import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from optuna import Study

from domain.models.config import (
    AppConfig,
    BacktestRequest,
    FitRequest,
    ForecasterType,
    TuneRequest,
)
from domain.models.interface import Forecaster
from features.dataset import DatasetLoader
from infrastructure.repository import BacktestRepository


class Forecasting:
    def __init__(
        self,
        loader: DatasetLoader,
        repository: BacktestRepository,
        path: Path,
        forecasters: list[Forecaster],
    ):
        self.loader = loader
        self.repository = repository
        self.path = path
        self.forecasters = forecasters

    def fit(self, config: AppConfig, request: FitRequest):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        for forecaster in self.forecasters:
            if request.forecaster and forecaster.name != request.forecaster:
                continue

            forecaster.load(self.path)

            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            forecaster.fit(df)
            forecaster.save(self.path)

    def backtest(self, config: AppConfig, request: BacktestRequest) -> pd.DataFrame:
        forecaster = self._get_forecaster(request.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        metric, result = forecaster.backtest(df)

        self.repository.save(result)

        return metric

    def tune(self, config: AppConfig, request: TuneRequest) -> Study:
        forecaster = self._get_forecaster(request.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        results, study = forecaster.tune(df, n_trials=request.trails)

        logging.debug(results)

        forecaster.save(self.path)

        return study

    def _get_forecaster(self, name: ForecasterType) -> Forecaster:
        for forecaster in self.forecasters:
            if forecaster.name == name:
                forecaster.load(self.path)
                return forecaster

        raise ValueError(f"Unknown forecaster: {name}")
