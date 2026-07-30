from dataclasses import dataclass
from enum import Enum

from domain.models.config import BacktestParams, FitParams, TuneParams


class JobType(str, Enum):
    FIT = "fit"
    TUNE = "tune"
    BACKTEST = "backtest"


@dataclass
class Job:
    id: str
    type: JobType
    params: FitParams | TuneParams | BacktestParams


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
