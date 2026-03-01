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
from context import HvacMode
from utils import generate_sunny_forecast

logger = logging.getLogger(__name__)


class Collector:
    def __init__(
        self, client: HAClient, database: Database, context: Context, config: Config
    ):
        self.weather = WeatherClient(client, context)
        self.client = client
        self.database = database
        self.context = context
        self.config = config

        self.current_slot_start = None

        self.pv_buffer = deque(maxlen=7)
        self.wp_buffer = deque(maxlen=7)
        self.grid_buffer = deque(maxlen=7)

        self.pv_slots = []
        self.wp_slots = []
        self.grid_slots = []
        self.supply_slots = []
        self.return_slots = []
        self.room_slots = []
        self.dhw_top_slots = []
        self.dhw_bottom_slots = []

    def update_forecast(self):
        self.client.reload()

        location = self.client.get_location()
        if location != (None, None):
            self.context.latitude, self.context.longitude = location
        else:
            logger.warning("[Collector] Locatie niet gevonden")

        solcast = self.client.get_forecast()

        now_local = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo)
        start_filter = now_local.replace(
            hour=8, minute=0, second=0, microsecond=0, day=14, month=1, year=2027
        ).tz_convert("UTC")
        end_filter = start_filter + timedelta(days=1)
        full_index = pd.date_range(
            start=start_filter, periods=193, freq="15min", name="timestamp"
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

        self.database.save_forecast(df_today)

        # forecast_df = df_today[df_today["timestamp"] >= self.context.now].copy()
        # self.context.forecast_df_raw = df_today
        self.context.forecast_df_raw = generate_sunny_forecast(self.context.now)

        logger.info("[Collector] Forecast updated")

    def update_load(self):
        self.client.reload()

        # 1. Haal ruwe waarden op
        raw_pv = self.client.get_pv_power()
        raw_wp = self.client.get_wp_power()
        raw_grid = self.client.get_grid_power()

        self._update_slot(self.pv_slots, raw_pv)
        self._update_slot(self.wp_slots, raw_wp)
        self._update_slot(self.grid_slots, raw_grid)

        # 2. Update de buffers en haal de mediaan op (filtert uitschieters/timing fouten)
        # We slaan de 'stable' waarden ook op in context voor debugging/UI
        self.context.stable_pv = self._update_buffer(self.pv_buffer, raw_pv)
        self.context.stable_wp = self._update_buffer(self.wp_buffer, raw_wp)
        self.context.stable_grid = self._update_buffer(self.grid_buffer, raw_grid)

        # 3. Berekening met gestabiliseerde waarden
        # Formule: Huisverbruik = (Netto Grid + PV Productie) - Warmtepomp
        total_consumption = self.context.stable_grid + self.context.stable_pv

        # Soms meten sensoren net iets anders (kalibratie).
        # Als WP zegt 2000W en Huis zegt 1950W, wordt base_load -50.
        base_load = total_consumption - self.context.stable_wp

        # 4. Intelligente fallback
        # Als base_load negatief is, is de meting van de WP waarschijnlijk hoger dan de P1/PV meten.
        # In dat geval is het 'restverbruik' van het huis waarschijnlijk minimaal.
        self.context.stable_load = max(0.05, base_load)

        logger.debug(
            f"[Collector] Load: Base={base_load:.2f}kW | "
            f"Calc: (Grid {self.context.stable_grid:.2f} + PV {self.context.stable_pv:.2f}) - WP {self.context.stable_wp:.2f}"
        )

    def update_sensors(self):
        raw_room = self.client.get_room_temp()
        raw_dhw_top = self.client.get_dhw_top()
        raw_dhw_bottom = self.client.get_dhw_bottom()

        self.context.room_temp = raw_room
        self.context.dhw_top = raw_dhw_top
        self.context.dhw_bottom = raw_dhw_bottom
        self.context.hvac_mode = self.client.get_hvac_mode()

        self._update_slot(self.room_slots, raw_room)
        self._update_slot(self.dhw_top_slots, raw_dhw_top)
        self._update_slot(self.dhw_bottom_slots, raw_dhw_bottom)

        if self.context.hvac_mode != HvacMode.OFF:
            self._update_slot(self.supply_slots, self.client.get_supply_temp())
            self._update_slot(self.return_slots, self.client.get_return_temp())

        logger.info("[Collector] Sensors updated")

    def update_history(self):
        """
        Berekent gemiddeld vermogen op basis van tellerstanden.
        """
        now = self.context.now
        aggregation_minutes = 15
        slot_minute = (now.minute // aggregation_minutes) * aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        # Initialisatie bij start applicatie
        if self.current_slot_start is None:
            self.current_slot_start = slot_start

            return

        # Detecteer kwartierwissel
        if slot_start > self.current_slot_start:
            avg_pv = self._mean(self.pv_slots)
            avg_wp = self._mean(self.wp_slots)
            avg_supply = self._mean(self.supply_slots)
            avg_return = self._mean(self.return_slots)
            avg_room = self._mean(self.room_slots)
            avg_dhw_top = self._mean(self.dhw_top_slots)
            avg_dhw_bottom = self._mean(self.dhw_bottom_slots)

            n_grid = len(self.grid_slots)
            if n_grid > 0:
                avg_import = sum(v for v in self.grid_slots if v > 0) / n_grid
                avg_export = (sum(v for v in self.grid_slots if v < 0) / n_grid) * -1.0
            else:
                avg_import = 0.0
                avg_export = 0.0

            self.pv_slots = []
            self.wp_slots = []
            self.grid_slots = []
            self.supply_slots = []
            self.return_slots = []
            self.room_slots = []
            self.dhw_top_slots = []
            self.dhw_bottom_slots = []

            # 4. Opslaan
            self.database.save_measurement(
                ts=self.current_slot_start,
                grid_import=avg_import,
                grid_export=avg_export,
                pv_actual=avg_pv,
                wp_actual=avg_wp,
                room_temp=avg_room,
                dhw_top=avg_dhw_top,
                dhw_bottom=avg_dhw_bottom,
                supply_temp=avg_supply,
                return_temp=avg_return,
                hvac_mode=int(self.context.hvac_mode.value),
            )

            self.current_slot_start = slot_start

            logger.info(
                f"[Collector] PV={avg_pv:.2f}kW WP={avg_wp:.2f}kW Grid={avg_import:.2f}/{avg_export:.2f}kW Room={avg_room:.2f}°C DHW={avg_dhw_top:.2f}/{avg_dhw_bottom:.2f}°C Supply={avg_supply:.2f}°C Return={avg_return:.2f}°C"
            )

    def _update_buffer(self, buffer: deque, value: float):
        if value is not None:
            buffer.append(value)

        if not buffer:
            return 0.0
        return float(np.median(buffer))

    def _update_slot(self, slot: list, value: float):
        if value is not None:
            slot.append(value)

    def _mean(self, value, default: float = np.nan):
        if not value:
            return default

        result = np.nanmean(value)

        if np.isnan(result):
            return default

        return float(result)

    def _median(self, value, default: float = np.nan):
        if not value:
            return default

        result = np.nanmedian(value)

        if np.isnan(result):
            return default

        return float(result)
