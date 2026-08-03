import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from domain.models.config import (
    BacktestParams,
    Config,
    FitParams,
    ForecasterType,
    PredictParams,
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

    def fit(self, config: Config, params: FitParams):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=params.days)

        for forecaster in self.forecasters:
            if params.forecaster and forecaster.name != params.forecaster:
                continue

            forecaster.load(self.path)

            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            forecaster.fit(df)

            forecaster.save(self.path)

    def predict(self, config: Config, params: PredictParams):
        forecaster = self._get_forecaster(params.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=1)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        result = forecaster.predict(df)

        print(result)

    def backtest(self, config: Config, params: BacktestParams):
        forecaster = self._get_forecaster(params.forecaster)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=params.days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        result = forecaster.backtest(df, steps=params.steps)

        logging.info(
            "Backtest finished: mae=%.3f",
            result.mae,
        )

        self.repository.save(result)

    def tune(self, config: Config, params: TuneParams):
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
            "Tune finished: mae=%.3f %s",
            study.best_value,
            " ".join(f"{key}={value}" for key, value in study.best_params.items()),
        )

        forecaster.save(self.path)

    def _get_forecaster(self, name: ForecasterType) -> Forecaster:
        for forecaster in self.forecasters:
            if forecaster.name == name:
                forecaster.load(self.path)
                return forecaster

        raise ValueError(f"Unknown forecaster: {name}")
