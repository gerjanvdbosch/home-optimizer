import logging

from dataclasses import dataclass
from context import Context
from config import Config
from optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    reason: str = None
    dhw_start_time: str = None


class Planner:
    def __init__(self, context: Context, config: Config):
        self.optimizer = Optimizer(config.pv_max_kw)
        self.context = context

    def create_plan(self):
        now = self.context.now

        outside_temp = 9

        dhw_profile = self.optimizer.calculate_profile(self.context.dhw_temp, self.context.dhw_setpoint, outside_temp)

        status, reason, solar_usage_kwh, current_load_val = self.optimizer.optimize(self.context.forecast_df, now, dhw_profile)

        logger.info(f"[Planner] Status={status} Reason={reason} BoilerSolar={solar_usage_kwh}kWh CurrentLoad={current_load_val}kW")

