import logging

from planner import Plan
from enum import Enum, auto
from statemachine import StateMachine

logger = logging.getLogger(__name__)


class DhwState(Enum):
    IDLE = auto()
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()


class DhwMachine(StateMachine):
    def __init__(self, context):
        super().__init__("DHW", context)
        self.state = DhwState.IDLE

    def on_enter(self, state):
        pass

    def process(self, plan: Plan):
        pass
