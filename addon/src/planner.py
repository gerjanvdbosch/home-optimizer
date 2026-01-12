import logging

from dataclasses import dataclass
from context import Context
from config import Config
from solar import SolarForecaster

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    dhw_start_time: str = None
    heating_start_time: str = None


class Planner:
    def __init__(self, context: Context, config: Config):
        self.forecaster = SolarForecaster(config, context)
        self.context = context

    def create_plan(self):
        now = self.context.now
        status, forecast = self.forecaster.analyze(now, self.context.stable_load)

        self.context.forecast = forecast
