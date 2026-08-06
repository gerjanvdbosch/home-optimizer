from dataclasses import dataclass
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
    id: str
    type: JobType
    config: FitConfig | PredictConfig | TuneConfig | BacktestConfig | Config


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
