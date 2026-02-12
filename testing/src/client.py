import logging
import json

from context import HvacMode
from config import Config

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self, config: Config):
        self.data = {}

    def reload(self):
        with open("data.json", "r") as f:
            self.data = json.load(f)

    def get_location(self):
        return 51.9, 5.3

    def get_pv_power(self):
        return float(self.data.get("pv_power", 0.0))

    def get_wp_power(self):
        return float(self.data.get("wp_power", 0.0))

    def get_grid_power(self):
        return float(self.data.get("grid_power", 0.0))

    def get_room_temp(self):
        return float(self.data.get("room_temp", 19.5))

    def get_dhw_top(self):
        return float(self.data.get("dhw_top", 30.0))

    def get_dhw_bottom(self):
        return float(self.data.get("dhw_bottom", 25.0))

    def get_dhw_setpoint(self):
        return float(self.data.get("dhw_setpoint", 50.0))

    def get_compressor_freq(self):
        return self.data.get("compressor_freq", 0.0)

    def get_supply_temp(self):
        return self.data.get("supply_temp", 0.0)

    def get_return_temp(self):
        return self.data.get("return_temp", 0.0)

    def get_hvac_mode(self):
        return HvacMode(HvacMode.OFF)

    def get_forecast(self):
        return self.data.get("solcast", {})

    def get_weather(self):
        return self.data.get("weather", {})
