import logging

from planner import Plan
from enum import Enum, auto
from statemachine import StateMachine

logger = logging.getLogger(__name__)


class ClimateState(Enum):
    NIGHT = auto()
    DAY_IDLE = auto()
    PAUSED_DHW = auto()


class ClimateMachine(StateMachine):
    def __init__(self, context):
        super().__init__("CLIMATE", context)
        self.state = ClimateState.DAY_IDLE

    def on_enter(self, state):
        pass

    def process(self, plan: Plan):
        pass
