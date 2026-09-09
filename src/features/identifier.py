import logging
from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
from joblib import dump, load

from domain.types import BacktestResult, Config
from features.dataset import DatasetDefinition

logger = logging.getLogger(__name__)

SystemModel = TypeVar("SystemModel")


class SystemIdentifier(Generic[SystemModel]):
    def __init__(self) -> None:
        self.model: SystemModel | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def label(self) -> str: ...

    @property
    @abstractmethod
    def unit(self) -> str: ...

    @abstractmethod
    def calibrate(self, df: pd.DataFrame) -> SystemModel: ...

    @abstractmethod
    def validate(
        self, df: pd.DataFrame, horizon_hours: float = 2.0
    ) -> BacktestResult: ...

    @abstractmethod
    def dataset(self, config: Config) -> DatasetDefinition: ...

    def get_model(self, dt_hours: float = 0.25) -> SystemModel:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Model is nog niet gekalibreerd.")
        if hasattr(self.model, "model_copy"):
            return self.model.model_copy(update={"dt_hours": dt_hours})
        return self.model

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Kan ongekalibreerd model niet opslaan.")
        path.mkdir(parents=True, exist_ok=True)
        target_file = path / f"{self.name}.joblib"
        dump(self.model, target_file)
        logger.info("[%s] Model succesvol opgeslagen naar %s", self.name, target_file)

    def load(self, path: Path) -> None:
        target_file = path / f"{self.name}.joblib"
        if not target_file.exists():
            logger.warning(
                "[%s] Geen opgeslagen model gevonden op %s",
                self.name,
                target_file,
            )
            return
        self.model = load(target_file)
        logger.info("[%s] Model succesvol geladen uit %s", self.name, target_file)
