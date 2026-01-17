import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from context import Context
from client import HAClient
from config import Config
from collections import deque
from weather import WeatherClient
from database import Database

logger = logging.getLogger(__name__)


class Collector:
    def __init__(
        self, client: HAClient, database: Database, context: Context, config: Config
    ):
        self.weather = WeatherClient(client)
        self.client = client
        self.database = database
        self.context = context
        self.config = config

    def update_forecast(self):
        self.client.reload()

        solcast = self.client.get_forecast(self.config.sensor_solcast)

        now_local = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo).replace(month=1, day=14, hour=10)
        start_filter = now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).tz_convert("UTC")
        end_filter = start_filter + timedelta(days=1)
        full_index = pd.date_range(
            start=start_filter, periods=97, freq="15min", name="timestamp"
        )

        df = pd.DataFrame(solcast)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)

        df_sol = (
            df.set_index("timestamp")
            .apply(pd.to_numeric, errors="coerce")
            .infer_objects(copy=False)
            .reindex(full_index)
            .interpolate(method="linear")
            .fillna(0)
            .reset_index()
        )

        df_om = self.weather.get_forecast()
        df_merged = pd.merge(df_sol, df_om, on="timestamp", how="left")

        df_today = (
            df_merged[
                (df_merged["timestamp"] >= start_filter)
                & (df_merged["timestamp"] <= end_filter)
            ]
            .copy()
            .sort_values("timestamp")
        )

        self.context.forecast_df = df_today
        self.database.save_forecast(df_today)

        logger.info("[Collector] Forecast updated")

    def update_sensors(self):
        self.client.reload()

        self.context.current_pv = self.client.get_pv_power(self.config.sensor_pv_power)
        self.context.current_wp = self.client.get_power_sensor(self.config.sensor_wp_power)
        self.context.current_grid = self.client.get_grid_power(self.config.sensor_grid_power)

        self.context.stable_pv = self._update_buffer(
            self.context.pv_buffer, self.context.current_pv
        )

        current_house_load = (self.context.current_grid + self.context.current_pv)
        self.context.stable_load = self._update_buffer(
            self.context.load_buffer, current_house_load
        )

        self.context.hvac_mode = self.client.get_hvac_mode(self.config.sensor_hvac)
        self.context.dhw_temp = self.client.get_dhw_temp(self.config.sensor_dhw_temp)
        self.context.dhw_setpoint = self.client.get_dhw_setpoint(self.config.sensor_dhw_setpoint)

        logger.info("[Collector] Sensors updated")

    def update_history(self):
        """
        Berekent gemiddeld vermogen op basis van tellerstanden.
        """
        now = self.context.now
        aggregation_minutes = 15
        slot_minute = (now.minute // aggregation_minutes) * aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        # 1. Haal HUIDIGE tellerstanden op
        curr_pv = self.client.get_state(self.config.sensor_pv_energy)
        curr_wp = self.client.get_state(self.config.sensor_wp_energy)
        curr_imp = self.client.get_state(self.config.sensor_grid_import_energy)
        curr_exp = self.client.get_state(self.config.sensor_grid_export_energy)

        # Initialisatie bij start applicatie
        if self.context.current_slot_start is None:
            self.context.current_slot_start = slot_start

            self.context.last_pv_kwh = curr_pv
            self.context.last_wp_kwh = curr_wp
            self.context.last_grid_import_kwh = curr_imp
            self.context.last_grid_export_kwh = curr_exp

            return

        # Detecteer kwartierwissel
        if slot_start > self.context.current_slot_start:
            # 2. Bereken vermogens t.o.v. VORIGE keer
            avg_pv = self._calculate_avg_power(curr_pv, self.context.last_pv_kwh)
            avg_wp = self._calculate_avg_power(curr_wp, self.context.last_wp_kwh)
            avg_imp = self._calculate_avg_power(curr_imp, self.context.last_grid_import_kwh)
            avg_exp = self._calculate_avg_power(curr_exp, self.context.last_grid_export_kwh)

            # 3. Update de 'last' waarden voor de volgende keer
            # Alleen updaten als we een geldige meting hebben
            if curr_pv is not None:
                self.context.last_pv_kwh = curr_pv
            if curr_wp is not None:
                self.context.last_wp_kwh = curr_wp
            if curr_imp is not None:
                self.context.last_grid_import_kwh = curr_imp
            if curr_exp is not None:
                self.context.last_grid_export_kwh = curr_exp

            # 4. Opslaan
            self.database.save_measurement(
                ts=self.context.current_slot_start
                grid_import=avg_imp,
                grid_export=avg_exp,
                pv_actual=avg_pv,
                wp_actual=avg_wp
            )

            self.context.current_slot_start = slot_start

            logger.info(
                f"[Collector] KWh-based Calc: PV:{avg_pv:.2f}kW | WP:{avg_wp:.2f}kW | Grid:{avg_imp:.2f}/{avg_exp:.2f}kW"
            )

    def _update_buffer(self, buffer: deque, value: float):
        if value is not None:
            buffer.append(value)

        if not buffer:
            return 0.0
        return float(np.median(buffer))

    # Helper functie voor de berekening
    def _calculate_avg_power(current, last):
        if current is None or last is None:
            return 0.0

        diff = current - last

        # Reset detectie: Als meter vervangen is of reset naar 0
        if diff < 0:
            return 0.0 # Of current * 4 als je aanneemt dat hij bij 0 begon

        # kWh naar kW (bij 15 min interval = maal 4)
        return diff * 4.0