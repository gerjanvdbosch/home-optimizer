import logging
from pathlib import Path

from app.state import StateManager
from domain.types import MPCConfig, MPCInput
from features.boiler import BoilerThermalIdentifier
from features.optimizer import MPCOptimizer
from infrastructure.repositories import ConfigRepository

logger = logging.getLogger(__name__)


class Optimization:
    def __init__(
        self,
        state_manager: StateManager,
        config_repository: ConfigRepository,
        models_path: Path,
    ) -> None:
        self.state_manager = state_manager
        self.config_repository = config_repository
        self.models_path = models_path

    def run(self) -> None:
        state = self.state_manager.load()
        config = self.config_repository.load()

        mpc_config = MPCConfig()

        solar_forecast = [p.value for p in state.predictions.solar]
        forecast_times = [p.time for p in state.predictions.solar]

        # The stored state.schedule.heat_pump.boiler.target_temperature is
        # resolved against *today's* timestamps (see StateManager.update()) - not
        # the future forecast horizon the MPC actually needs. Resolve the raw
        # config schedule against the forecast's own timestamps instead.
        target_temps = tuple(
            point.value
            for point in self.state_manager.resolve_schedule(
                config.heat_pump.boiler.target_temperature, forecast_times
            )
        )

        dynamics_forecaster = BoilerThermalIdentifier()
        dynamics_forecaster.load(path=self.models_path)
        thermal_model = dynamics_forecaster.get_model()

        heat_pump_state = state.measurements.heat_pump.state
        boiler_on_current = bool(heat_pump_state) and (
            heat_pump_state[-1].value == BoilerThermalIdentifier.DHW_ACTIVE_STATE
        )

        # Aligned against solar's own forecast timestamps, not assumed to share
        # them: the tap forecaster is fit/predicted independently (see
        # features/tap.py) and may not have been run at all, or over a
        # different horizon - align_predictions falls back to 0.0 (no draws
        # assumed) wherever no matching point exists, the same assumption
        # implicitly made before this forecast existed.
        tap_forecast = tuple(
            self.state_manager.align_predictions(state.predictions.tap, forecast_times)
        )

        data = MPCInput(
            solar_forecast_w=solar_forecast,
            ambient_temperature=state.measurements.heat_pump.boiler.ambient_temperature[
                -1
            ].value,
            current_temp_top=state.measurements.heat_pump.boiler.top_temperature[
                -1
            ].value,
            current_temp_bottom=state.measurements.heat_pump.boiler.bottom_temperature[
                -1
            ].value,
            boiler_on_current=boiler_on_current,
            thermal_model=thermal_model,
            target_temperature_top=target_temps,
            tap_forecast_w=tap_forecast,
        )

        optimizer = MPCOptimizer(thermal_model=thermal_model, config=mpc_config)
        result = optimizer.solve(data)

        logger.info(
            "Optimization completed: schedule=%s objective=%.3f",
            result.schedule,
            result.objective_value,
        )

        self.state_manager.update_schedule(
            schedule=result.schedule,
            temperatures=result.temperatures,
            power_w=mpc_config.boiler_electrical_power_w,
            times=forecast_times,
        )
