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
        Verzamelt samples en schrijft elke 15 minuten naar de database.
        """
        now = self.context.now
        aggregation_minutes = 15
        slot_minute = (now.minute // aggregation_minutes) * aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        # Initialisatie bij start applicatie
        if self.context.current_slot_start is None:
            self.context.current_slot_start = slot_start

        # Detecteer kwartierwissel
        if slot_start > self.context.current_slot_start:
            # 1. Bereken gemiddelden (gebruik nanmean voor robuustheid tegen dropouts)
            avg_pv = float(np.nanmean(self.context.pv_samples)) if self.context.pv_samples else 0.0
            avg_wp = float(np.nanmean(self.context.wp_samples)) if self.context.wp_samples else 0.0
            avg_grid_raw = float(np.nanmean(self.context.grid_samples)) if self.context.grid_samples else 0.0

            # 2. Splits Grid in Import en Export
            # Import = Alles boven 0
            # Export = Alles onder 0 (positief gemaakt)
            grid_import = max(0.0, avg_grid_raw)
            grid_export = abs(min(0.0, avg_grid_raw))

            # 3. Sla op in de nieuwe 'Measurements' tabel
            self.database.save_measurement(
                ts=timestamp,
                grid_import=grid_import,
                grid_export=grid_export,
                pv_actual=avg_pv,
                wp_actual=avg_wp
            )

            logger.info(
                f"[Collector] History saved {timestamp:%H:%M} | "
                f"Imp: {grid_import:.2f} | Exp: {grid_export:.2f} | "
                f"PV: {avg_pv:.2f} | WP: {avg_wp:.2f}"
            )

            self.context.pv_samples = []
            self.context.wp_samples = []
            self.context.grid_samples = []
            self.context.current_slot_start = slot_start

        # Voeg huidige metingen toe aan de buffers (voor gemiddelde berekening later)
        if self.context.current_pv is not None:
            self.context.pv_samples.append(self.context.current_pv)

        if self.context.current_wp is not None:
            self.context.wp_samples.append(self.context.current_wp)

        if self.context.current_grid is not None:
            self.context.grid_samples.append(self.context.current_grid)


    def _update_buffer(self, buffer: deque, value: float):
        if value is not None:
            buffer.append(value)

        if not buffer:
            return 0.0
        return float(np.median(buffer))
