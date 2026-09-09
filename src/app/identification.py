import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.types import BacktestConfig, CalibrateConfig
from features.dataset import DatasetLoader
from features.identifier import SystemIdentifier
from infrastructure.repositories import BacktestRepository, ConfigRepository

logger = logging.getLogger(__name__)


class Identification:
    def __init__(
        self,
        loader: DatasetLoader,
        backtest_repository: BacktestRepository,
        config_repository: ConfigRepository,
        state_manager: Any,
        path: Path,
        identifiers: list[SystemIdentifier],
    ):
        self.loader = loader
        self.backtest_repository = backtest_repository
        self.config_repository = config_repository
        self.state_manager = state_manager
        self.path = path
        self.identifiers = identifiers

    def calibrate(self, config: CalibrateConfig) -> None:
        for identifier in self.identifiers:
            if config.target and identifier.name != config.target:
                continue

            identifier, df = self._prepare(identifier, config.days)

            # 1. Parameteridentificatie uitvoeren
            plant_model = identifier.calibrate(df)

            # 2. Model opslaan naar disk
            identifier.save(self.path)

            logger.info("[%s] Kalibratie succesvol afgerond.", identifier.name)

    def validate(self, config: BacktestConfig) -> None:
        """Voert een rolling-horizon validatie uit en slaat het BacktestResult op."""
        identifier, df = self._prepare(config.target, config.days)

        # Validatie uitvoeren (standaard horizon van 2 uur tegen tapwaterdrift)
        horizon_hours = getattr(config, "horizon_hours", 2.0)
        result = identifier.validate(df, horizon_hours=horizon_hours)

        logger.info(
            "[%s] Validation finished: mae=%.3f, rmse=%.3f",
            result.name,
            result.mae,
            result.rmse,
        )

        # Direct opslaan in de backtest repo (werkt 1-op-1 met je backtest_chart!)
        self.backtest_repository.save(result)

    def get_model(self, name: str, dt_hours: float = 0.25) -> Any:
        """Haalt het actuele gekalibreerde plant model op t.b.v. de MPC optimizer."""
        identifier = self._get_identifier(name)
        identifier.load(self.path)
        return identifier.get_model(dt_hours=dt_hours)

    def _prepare(
        self,
        identifier: str | SystemIdentifier,
        days: int,
    ) -> tuple[SystemIdentifier, Any]:
        """Laadt het bestaande model en haalt de benodigde dataset op."""
        if isinstance(identifier, str):
            identifier = self._get_identifier(identifier)

        identifier.load(self.path)

        config = self.config_repository.load()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        dataset = identifier.dataset(config)
        df = self.loader.load(dataset, start, end)

        return identifier, df

    def _get_identifier(self, name: str) -> SystemIdentifier:
        for identifier in self.identifiers:
            if identifier.name == name:
                return identifier

        raise ValueError(f"Unknown system identifier: {name}")
