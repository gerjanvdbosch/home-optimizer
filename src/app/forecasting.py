import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.models.config import (
    BacktestConfig,
    Config,
    FitConfig,
    ForecasterType,
    PredictConfig,
    TuneConfig,
)
from domain.models.interface import Forecaster
from domain.models.state import SeriesPoint
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
        forecasters: list[Forecaster],
    ):
        self.loader = loader
        self.backtest_repository = backtest_repository
        self.config_repository = config_repository
        self.state_manager = state_manager
        self.path = path
        self.forecasters = forecasters

    def update(self, config: Config):
        self.config_repository.save(config)
        self.state_manager.update(config)
        self.backtest_repository.clear()

    def fit(self, config: FitConfig):
        for forecaster in self.forecasters:
            if config.forecaster and forecaster.name != config.forecaster:
                continue

            forecaster, df = self._prepare(forecaster, config.days)

            forecaster.fit(df)

            forecaster.save(self.path)

    def predict(self, config: PredictConfig):
        forecaster, df = self._prepare(config.forecaster, 1)

        now = datetime.now(timezone.utc)

        last_window = df[
            (df["target_time"] <= now) & df["P_solar"].notna()
        ].sort_values("target_time")

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
        )

        print(last_window)
        print(future)

        result = forecaster.predict(
            last_window=last_window,
            df=future,
            steps=config.steps,
        )

        print(result)

        # state = self.state_manager.load()
        #
        # points = [
        #     SeriesPoint(time=pd.to_datetime(str(time)), value=float(value))
        #     for time, value in result.items()
        # ]
        #
        # if forecaster.name == "solar":
        #     state.forecast.solar.predicted = points

        # self.state_manager.save(state)

    def backtest(self, config: BacktestConfig):
        forecaster, df = self._prepare(config.forecaster, config.days)

        result = forecaster.backtest(df, steps=config.steps)

        logging.info(
            "Backtest finished: mae=%.3f",
            result.mae,
        )

        self.backtest_repository.save(result)

    def tune(self, config: TuneConfig):
        forecaster, df = self._prepare(config.forecaster, config.days)

        results, study = forecaster.tune(
            df,
            n_trials=config.trails,
            study_storage=f"sqlite:///{self.path / 'optuna.db'}",
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

        forecaster.load(self.path)

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
