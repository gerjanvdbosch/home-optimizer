import logging

from dataclasses import dataclass
from context import Context
from config import Config
from solar import SolarForecaster
from datetime import datetime, timezone

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
        logger.info(f"[Planner] Status {status}")

        if forecast is not None:
            if forecast.planned_start is None:
                forecast.planned_start = datetime.now(timezone.utc).replace(
                    hour=16, minute=0, second=0, microsecond=0
                )

                if forecast.planned_start < now:
                    forecast.planned_start = None


#             logger.info(f"[Planner] Reason {forecast.reason}")
#             logger.info(f"[Planner] PV now {forecast.actual_pv}kW")
#             logger.info(f"[Planner] Load now {forecast.load_now}kW")
#             logger.info(f"[Planner] Forecast now {forecast.energy_now}kW")
#             logger.info(f"[Planner] Forecast best {forecast.energy_best}kW")
#             logger.info(f"[Planner] Opportunity cost {forecast.opportunity_cost}")
#             logger.info(f"[Planner] Confidence {forecast.confidence}")
#             logger.info(f"[Planner] Bias {forecast.current_bias}")
#             logger.info(f"[Planner] Planned start {forecast.planned_start}")

# Compressor freq gebruiken voor load / power inschatting

# Planner wanneer de verwarming aan moet als het x tijd warm moet zijn

# In zomer, beste DHW moment plannen
# In winter, beste verwarmingsmoment plannen en evt in piek/teruglevering DHW (verwarmen stoppen)
# DHW zo laat mogelijk voor deadline plannen

# Als het warm blijft tot x uur, geen verwarming nodig en lager zetten


#     def create_plan(self) -> Plan:
#         now = datetime.now(timezone.utc)
#         current_load = self.context.current_load if self.context.current_load else 0.0
#
#         # 1. DE ANALYSE (Alles in één keer: Predict -> NowCast -> Optimize)
#         # We krijgen direct de beste starttijd terug, gebaseerd op de live situatie.
#         solar_status, solar_context = self.forecaster.analyze(now, current_load)
#
#         if solar_context is None:
#             return Plan("OFF", "Geen solar data")
#
#         # 2. MAAK HET SCHEMA (MPC)
#         # We geven de berekende starttijd mee aan de scheduler
#         df_plan = self.calculate_mpc_schedule(
#             self.context.forecast_df,
#             boiler_start_time=solar_context.planned_start,
#             is_dark_day=(solar_context.energy_best < 0.5) # Of een andere indicator uit context
#         )
#
#         self.context.forecast_df = df_plan # Opslaan voor grafiek
#
#         # 3. BEPAAL ACTIE VOOR NU
#         now_utc = pd.Timestamp.now(tz="UTC").floor("15min")
#         if now_utc in df_plan["timestamp"].values:
#             row = df_plan.loc[df_plan["timestamp"] == now_utc].iloc[0]
#             return Plan(row["plan"], row.get("reason", "Volgens schema"))
#
#         return Plan("OFF", "Buiten bereik")
#
#       def calculate_mpc_schedule(self, df: pd.DataFrame, boiler_start_time, is_dark_day: bool) -> pd.DataFrame:
#            df = df.copy()
#            df["plan"] = "OFF"
#            df["reason"] = ""
#            df["plan_power"] = 0.0
#
#            now = datetime.now(timezone.utc)
#            current_temp = getattr(self.context, 'room_temp', 19.0)
#
#            # --- STAP A: BOILER PLANNEN (Gebruik de tijd uit de analyse!) ---
#            if boiler_start_time:
#                # Zoek index van de starttijd
#                matches = df.index[df['timestamp'] == boiler_start_time]
#                if len(matches) > 0:
#                    start_pos = matches[0]
#                    # Boiler duur uit config halen (bijv. 1.5 uur = 6 kwartieren)
#                    steps = int(self.config.dhw_duration_hours * 4)
#                    end_pos = min(start_pos + steps, len(df))
#
#                    df.iloc[start_pos:end_pos, df.columns.get_loc("plan")] = "BOILER"
#                    df.iloc[start_pos:end_pos, df.columns.get_loc("plan_power")] = 2.5
#                    df.iloc[start_pos:end_pos, df.columns.get_loc("reason")] = "Solar Piek" if not is_dark_day else "Dark Day COP"
#
#         # --- STAP B: MPC VERWARMING (Pre-heat/Post-heat) ---
#         # We kijken 12 uur vooruit naar deadlines
#         horizon = df[df['timestamp'] >= now].iloc[:48]
#         start_mpc_heating = False
#         mpc_reason = ""
#
#         for _, row in horizon.iterrows():
#             sim_time = row['timestamp']
#             local_time = sim_time.astimezone().time()
#             target_temp = get_target_temp(local_time)
#
#             if target_temp > current_temp:
#                 mins_needed = self.thermal_model.calculate_time_to_heat(...)
#                 mins_avail = (sim_time - now).total_seconds() / 60
#
#                 # Check boiler conflict in het DataFrame
#                 conflict_rows = df[
#                     (df['timestamp'] >= now) &
#                     (df['timestamp'] < sim_time) &
#                     (df['plan'] == "BOILER")
#                 ]
#                 loss_mins = len(conflict_rows) * 15
#
#                 if mins_needed >= (mins_avail - loss_mins - 30):
#                     start_mpc_heating = True
#                     mpc_reason = f"Pre-heat voor {local_time.strftime('%H:%M')}"
#                     break
#
#         # --- STAP C: INVULLEN ---
#         # Belangrijk: Overschrijf de Boiler NOOIT
#         if start_mpc_heating:
#             if df.iloc[0]['plan'] == "OFF":
#                 df.iloc[0, df.columns.get_loc("plan")] = "HEAT_PUMP"
#                 df.iloc[0, df.columns.get_loc("reason")] = mpc_reason
#
#         # 2. Solar Bonus (Bufferen)
#         # Als MPC niet hoeft, maar er is wel veel zon over -> Toch verwarmen
#         elif df.iloc[0]['plan'] == "OFF":
#             surplus = df.iloc[0]['power_corrected'] - df.iloc[0]['consumption']
#             if surplus > 0.8: # 800W overschot
#                 df.iloc[0, df.columns.get_loc("plan")] = "HEAT_PUMP"
#                 df.iloc[0, df.columns.get_loc("reason")] = "Solar Buffer"
#
#         return df
