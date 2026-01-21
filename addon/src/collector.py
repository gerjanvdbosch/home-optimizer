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
        self.weather = WeatherClient(config, context)
        self.client = client
        self.database = database
        self.context = context
        self.config = config

    def update_forecast(self):
        solcast = self.client.get_forecast()

        now_local = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo)
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

    def update_load(self):
        # 1. Haal ruwe waarden op
        raw_pv = self.client.get_pv_power()  # Zorg dat dit altijd >= 0 is
        raw_wp = self.client.get_wp_power()  # Zorg dat dit altijd >= 0 is
        raw_grid = (
            self.client.get_grid_power()
        )  # BELANGRIJK: Import - Export (kan negatief zijn)

        # 2. Update de buffers en haal de mediaan op (filtert uitschieters/timing fouten)
        # We slaan de 'stable' waarden ook op in context voor debugging/UI
        self.context.current_pv = self._update_buffer(self.context.pv_buffer, raw_pv)
        self.context.current_wp = self._update_buffer(self.context.wp_buffer, raw_wp)
        self.context.current_grid = self._update_buffer(
            self.context.grid_buffer, raw_grid
        )

        # 3. Berekening met gestabiliseerde waarden
        # Formule: Huisverbruik = (Netto Grid + PV Productie) - Warmtepomp
        total_consumption = self.context.current_grid + self.context.current_pv

        # Soms meten sensoren net iets anders (kalibratie).
        # Als WP zegt 2000W en Huis zegt 1950W, wordt base_load -50.
        base_load = total_consumption - self.context.current_wp

        # 4. Intelligente fallback
        # Als base_load negatief is, is de meting van de WP waarschijnlijk hoger dan de P1/PV meten.
        # In dat geval is het 'restverbruik' van het huis waarschijnlijk minimaal.
        self.context.stable_load = max(0.1, base_load)

        logger.debug(
            f"[Collector] Load: Base={base_load:.2f}kW | "
            f"Calc: (Grid {self.context.current_grid:.2f} + PV {self.context.current_pv:.2f}) - WP {self.context.current_wp:.2f}"
        )

    def update_sensors(self):
        location = self.client.get_location()
        if location != (None, None):
            self.context.latitude, self.context.longitude = location
        else:
            logger.warning("[Collector] Locatie niet gevonden")

        self.context.hvac_mode = self.client.get_hvac_mode()
        self.context.dhw_temp = self.client.get_dhw_temp()
        self.context.dhw_setpoint = self.client.get_dhw_setpoint()

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
        current_pv = self.client.get_pv_energy()
        current_wp = self.client.get_wp_energy()
        current_import = self.client.get_grid_import()
        current_export = self.client.get_grid_export()

        # Initialisatie bij start applicatie
        if self.context.current_slot_start is None:
            self.context.current_slot_start = slot_start

            self.context.last_pv = current_pv
            self.context.last_wp = current_wp
            self.context.last_grid_import = current_import
            self.context.last_grid_export = current_export

            return

        # Detecteer kwartierwissel
        if slot_start > self.context.current_slot_start:
            # 2. Bereken vermogens t.o.v. VORIGE keer
            avg_pv = self._calculate_avg_power(current_pv, self.context.last_pv)
            avg_wp = self._calculate_avg_power(current_wp, self.context.last_wp)
            avg_import = self._calculate_avg_power(
                current_import, self.context.last_grid_import
            )
            avg_export = self._calculate_avg_power(
                current_export, self.context.last_grid_export
            )

            # 3. Update de 'last' waarden voor de volgende keer
            # Alleen updaten als we een geldige meting hebben
            if current_pv is not None:
                self.context.last_pv = current_pv
            if current_wp is not None:
                self.context.last_wp = current_wp
            if current_import is not None:
                self.context.last_grid_import = current_import
            if current_export is not None:
                self.context.last_grid_export = current_export

            # 4. Opslaan
            self.database.save_measurement(
                ts=self.context.current_slot_start,
                grid_import=avg_import,
                grid_export=avg_export,
                pv_actual=avg_pv,
                wp_actual=avg_wp,
            )

            self.context.current_slot_start = slot_start

            logger.info(
                f"[Collector] PV={avg_pv:.2f}kW WP={avg_wp:.2f}kW Grid={avg_import:.2f}/{avg_export:.2f}kW"
            )

    def _update_buffer(self, buffer: deque, value: float):
        if value is not None:
            buffer.append(value)

        if not buffer:
            return 0.0
        return float(np.median(buffer))

    # Helper functie voor de berekening
    def _calculate_avg_power(self, current, last):
        if current is None or last is None:
            return 0.0

        diff = current - last

        # Reset detectie: Als meter vervangen is of reset naar 0
        if diff < 0:
            return 0.0  # Of current * 4 als je aanneemt dat hij bij 0 begon

        # kWh naar kW (bij 15 min interval = maal 4)
        return diff * 4.0
