import logging

from app.state import StateManager
from domain.models import MPCConfig, MPCInput
from features.estimator import DisturbanceEstimator
from features.forecasters.boiler import BoilerForecaster
from features.optimizer import MPCOptimizer

logger = logging.getLogger(__name__)


class Optimization:
    def __init__(self, state_manager: StateManager, models_path) -> None:
        self.state_manager = state_manager
        self.models_path = models_path

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

        target_temps = [
            p.value for p in state.schedule.heat_pump.boiler.target_temperature
        ]

        dynamics_forecaster = BoilerForecaster()
        dynamics_forecaster.load(path=self.models_path)
        thermal_model = dynamics_forecaster.to_thermal_model()

        print(thermal_model)

        data = MPCInput(
            solar_forecast_kw=solar_trajectory,
            boiler_on=False,
            # current_temp_top=state.measurements.heat_pump.boiler.top_temperature[
            #     -1
            # ].value,
            # current_temp_bottom=state.measurements.heat_pump.boiler.bottom_temperature[
            #     -1
            # ].value,
            current_temp_top=44,
            current_temp_bottom=27,
            thermal_model=thermal_model,
            target_temperature_top=target_temps,
            ambient_temperature=state.measurements.heat_pump.boiler.ambient_temperature[
                -1
            ].value,
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
            temperatures_top=result.temperatures_top,
            temperatures_bottom=result.temperatures_bottom,
            power_kw=mpc_config.boiler_power,
            times=[p.time for p in state.predictions.solar],
        )
