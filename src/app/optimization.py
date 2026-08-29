import logging

from app.state import StateManager
from domain.models import MPCConfig, MPCInput
from features.estimator import DisturbanceEstimator
from features.optimizer import MPCOptimizer

logger = logging.getLogger(__name__)


class Optimization:
    def __init__(self, state_manager: StateManager) -> None:
        self.state_manager = state_manager

    def run(self) -> None:
        state = self.state_manager.load()

        mpc_config = MPCConfig(
            step_hours=0.25,
            boiler_power=2500.0,
            boiler_duration_hours=1.0,
        )

        solar_forecast = [p.value for p in state.predictions.solar]
        actual_now = (
            state.measurements.solar[-1].value if state.measurements.solar else None
        )

        print(actual_now)

        disturbance_estimator = DisturbanceEstimator(
            alpha=0.7,
            max=mpc_config.boiler_power,
        )
        solar_trajectory = disturbance_estimator.estimate(
            forecast=solar_forecast,
            actual_now=actual_now,
        )

        data = MPCInput(
            solar_forecast_kw=solar_trajectory,
            boiler_on=False,
        )

        optimizer = MPCOptimizer(mpc_config)
        result = optimizer.solve(data)

        logger.info(
            "Optimization completed: schedule=%s objective=%.3f",
            result.schedule,
            result.objective_value,
        )

        self.state_manager.update_schedule(
            schedule=result.schedule,
            power_kw=mpc_config.boiler_power,
            times=[p.time for p in state.predictions.solar],
        )
