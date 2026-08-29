import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.models import (
    BacktestConfig,
    Config,
    FitConfig,
    Forecaster,
    ForecasterType,
    PredictConfig,
    TuneConfig,
)
from features.dataset import DatasetLoader
from infrastructure.repository import BacktestRepository, ConfigRepository


class Forecasting:
    def __init__(
        self,
        loader: DatasetLoader,
        backtest_repository: BacktestRepository,
        config_repository: ConfigRepository,
        state_manager: Any,
        path: Path,
        study_storage: str,
        forecasters: list[Forecaster],
    ):
        self.loader = loader
        self.backtest_repository = backtest_repository
        self.config_repository = config_repository
        self.state_manager = state_manager
        self.path = path
        self.study_storage = study_storage
        self.forecasters = forecasters

    def update(self, config: Config) -> None:
        self.config_repository.save(config)
        self.state_manager.update(config)
        self.backtest_repository.clear()

    def fit(self, config: FitConfig) -> None:
        for forecaster in self.forecasters:
            if config.forecaster and forecaster.name != config.forecaster:
                continue

            forecaster, df = self._prepare(forecaster, config.days)

            forecaster.fit(df)

            forecaster.save(self.path)

    def predict(self, config: PredictConfig) -> None:
        forecaster, df = self._prepare(config.forecaster, 7)

        result = forecaster.predict(df=df, steps=config.steps)

        self.state_manager.update_prediction(forecaster.name, result)

    def backtest(self, config: BacktestConfig) -> None:
        forecaster, df = self._prepare(config.forecaster, config.days)

        result = forecaster.backtest(df, steps=config.steps)

        logging.info(
            "Backtest finished: mae=%.3f",
            result.mae,
        )

        self.backtest_repository.save(result)

    def tune(self, config: TuneConfig) -> None:
        forecaster, df = self._prepare(config.forecaster, config.days)

        self.path.mkdir(parents=True, exist_ok=True)

        results, study = forecaster.tune(
            df,
            n_trials=config.trails,
            study_storage=self.study_storage,
        )

        logging.info(
            "Tune finished: mae=%.3f %s",
            study.best_value,
            " ".join(f"{key}={value}" for key, value in study.best_params.items()),
        )

        forecaster.save(self.path)

    def _prepare(
        self,
        forecaster: ForecasterType | Forecaster,
        days: int,
    ) -> tuple[Forecaster, Any]:
        if isinstance(forecaster, str):
            forecaster = self._get_forecaster(forecaster)

        forecaster.load(self.path, self.study_storage)

        config = self.config_repository.load()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        dataset = forecaster.dataset(config)

        df = self.loader.load(dataset, start, end)

        return forecaster, df

    def _get_forecaster(self, name: ForecasterType) -> Forecaster:
        for forecaster in self.forecasters:
            if forecaster.name == name:
                return forecaster

        raise ValueError(f"Unknown forecaster: {name}")
