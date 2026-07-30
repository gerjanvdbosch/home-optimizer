from dataclasses import dataclass
from enum import Enum

from domain.models.config import BacktestRequest, FitRequest, TuneRequest


class JobType(str, Enum):
    FIT = "fit"
    TUNE = "tune"
    BACKTEST = "backtest"


@dataclass
class Job:
    id: str
    type: JobType
    request: FitRequest | TuneRequest | BacktestRequest


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
