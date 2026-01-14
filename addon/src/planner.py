import logging

from dataclasses import dataclass
from context import Context
from config import Config
from solar import SolarForecaster
from optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None


class Planner:
    def __init__(self, context: Context, config: Config):
        self.forecaster = SolarForecaster(config, context)
        self.context = context
        self.config = config

    def create_plan(self):
        now = self.context.now
        status, forecast = self.forecaster.analyze(now, self.context.stable_load)

        self.context.forecast = forecast

        # 1. Meet je sensoren
        current_water_temp = self.context.dhw_temp  # Sensor
        target_water_temp = 50.0
        outside_temp = 7.0  # API of sensor

        # 2. Initialiseer optimizer (zonder vaste duur, die is nu dynamisch)
        opt = Optimizer(self.config.pv_max_kw)

        # 3. Bereken het profiel
        # Dit geeft bijv. [1.7, 2.0, 2.4, 2.7, 2.7] terug als er veel energie nodig is
        profile = opt.calculate_profile(
            current_water_temp, target_water_temp, outside_temp=outside_temp
        )

        # 4. Optimaliseer
        status, context = opt.optimize(self.context.forecast_df, now, profile)

        logger.info(
            f"[Planner] Status {status}, Reason: {context.reason}, Energy Best: {context.energy_best}kWh"
        )


#         logger.info(f"[Planner] Status {status}")
#
#         if forecast is not None:
#             if forecast.planned_start is None:
#                 forecast.planned_start = datetime.now(timezone.utc).replace(
#                     hour=16, minute=0, second=0, microsecond=0
#                 )
#
#                 if forecast.planned_start < now:
#                     forecast.planned_start = None

#             logger.info(f"[Planner] Reason {forecast.reason}")
#             logger.info(f"[Planner] PV now {forecast.actual_pv}kW")
#             logger.info(f"[Planner] Load now {forecast.load_now}kW")
#             logger.info(f"[Planner] Forecast now {forecast.energy_now}kW")
#             logger.info(f"[Planner] Forecast best {forecast.energy_best}kW")
#             logger.info(f"[Planner] Opportunity cost {forecast.opportunity_cost}")
#             logger.info(f"[Planner] Confidence {forecast.confidence}")
#             logger.info(f"[Planner] Bias {forecast.current_bias}")
#             logger.info(f"[Planner] Planned start {forecast.planned_start}")
