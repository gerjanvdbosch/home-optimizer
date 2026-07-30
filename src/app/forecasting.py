import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from optuna import Study

from domain.models.config import (
    BacktestParams,
    Config,
    FitParams,
    ForecasterType,
    TuneParams,
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

    def predict(self, config: Config):
        pass

    def fit(self, config: Config, params: FitParams):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=params.days)

        for forecaster in self.forecasters:
            if params.forecaster and forecaster.name != params.forecaster:
                continue

            forecaster.load(self.path)

            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            logging.info("Fit finished: ")

            forecaster.fit(df)
            forecaster.save(self.path)

    def backtest(self, config: Config, params: BacktestParams) -> pd.DataFrame:
        forecaster = self._get_forecaster(params.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=params.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        metric, result = forecaster.backtest(df)

        logging.info(
            "Backtest finished: metric=%s",
            metric,
        )

        self.repository.save(result)

        return metric

    def tune(self, config: Config, params: TuneParams) -> Study:
        forecaster = self._get_forecaster(params.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=params.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        results, study = forecaster.tune(
            df,
            n_trials=params.trails,
            study_storage=f"sqlite:///{self.path / 'optuna.db'}",
        )

        logging.info(
            "Tune finished: value=%.3f params=%s",
            study.best_value,
            study.best_params,
        )

        forecaster.save(self.path)

        return study

    def _get_forecaster(self, name: ForecasterType) -> Forecaster:
        for forecaster in self.forecasters:
            if forecaster.name == name:
                forecaster.load(self.path)
                return forecaster

        raise ValueError(f"Unknown forecaster: {name}")
