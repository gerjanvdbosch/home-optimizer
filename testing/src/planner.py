import logging

from dataclasses import dataclass
from context import Context
from config import Config
from solar import SolarForecaster
from mpc import BoilerMPC, BoilerConfig

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str
    reason: str
    mpc_power: float = 0.0

class Planner:
    def __init__(self, context: Context, config: Config):
        self.forecaster = SolarForecaster(config, context)
        self.context = context
        mpc_cfg = BoilerConfig(
            volume_liters=200,
            power_kw=2.2,
            deadline_hour=17
        )
        self.mpc = BoilerMPC(mpc_cfg)

    def create_plan(self):
        now = self.context.now
        status, forecast = self.forecaster.analyze(now, self.context.stable_load)

        self.context.forecast = forecast

        df_future = self.context.forecast_df[
            self.context.forecast_df['timestamp'] >= now
        ].copy()

        temp_now = self.context.dhw_temp
        load_now = self.context.stable_load

        # 2. Draai de MPC Solver
        result_df = self.mpc.solve(df_future, temp_now, load_now)

        if result_df is None:
            return Plan("IDLE", "MPC Error (Infeasible)")

        # 3. Lees de EERSTE stap uit het plan
        # Wat moet er NU (komende 15 min) gebeuren?
        power_cmd = result_df.iloc[0]["mpc_power_kw"]
        predicted_temp = result_df.iloc[0]["mpc_temp"]

        # Drempelwaarde voor relais (b.v. 500W)
        should_heat = power_cmd > 1.0

        if should_heat:
            is_solar = result_df.iloc[0]["power_corrected"] > (self.context.stable_load + 1.0)
            reason_src = "Zonnestroom" if is_solar else "Deadline/Grid"

            # GEBRUIK HIER predicted_temp IN DE TEKST:
            reason = f"{reason_src} (MPC: {power_cmd:.2f}kW -> {predicted_temp:.1f}°C)"

            return Plan("HEAT_DHW", reason, power_cmd)

        else:
            # Ook als we niks doen, is het leuk om te weten wat de temp gaat doen (b.v. dalen)
            future_starts = result_df[result_df["boiler_status"] == 1]
            if not future_starts.empty:
                next_time = future_starts.iloc[0]["timestamp"].strftime("%H:%M")
                msg = f"Wacht op start {next_time} (Verwacht: {predicted_temp:.1f}°C)"
            else:
                msg = f"In rust (Verwacht: {predicted_temp:.1f}°C)"

            return Plan("IDLE", msg, power_cmd)

