import uuid
from dataclasses import dataclass, field
from enum import Enum

from domain.models.config import (
    BacktestConfig,
    Config,
    FitConfig,
    PredictConfig,
    TuneConfig,
)


class JobType(str, Enum):
    UPDATE = "update"
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"


@dataclass
class Job:
    type: JobType
    config: FitConfig | PredictConfig | TuneConfig | BacktestConfig | Config
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
