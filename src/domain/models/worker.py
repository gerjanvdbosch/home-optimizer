from dataclasses import dataclass
from enum import Enum

from domain.models.config import BacktestParams, FitParams, PredictParams, TuneParams


class JobType(str, Enum):
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"


@dataclass
class Job:
    id: str
    type: JobType
    params: FitParams | PredictParams | TuneParams | BacktestParams


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
