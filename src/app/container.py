from dataclasses import dataclass

from app.state_update import StateUpdater
from app.trainer import Trainer


@dataclass(slots=True)
class Container:
    state_updater: StateUpdater
    trainer: Trainer
