import os
import requests
import logging
import json

from context import HvacMode

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self):
        self.data = {}

    def reload(self):
        with open("data.json", "r") as f:
            self.data = json.load(f)

    def get_location(self, entity_id):
        return 51.9, 5.3

    def get_pv_power(self, entity_id):
        return float(self.data.get("pv_power", 0.0))

    def get_load_power(self, entity_id):
        return float(self.data.get("load_power", 0.0))

    def get_room_temp(self):
        return float(self.data.get("room_temp", 19.5))

    def get_dhw_temp(self, entity_id):
        return float(self.data.get("dhw_temp", 30.0))

    def get_dhw_setpoint(self, entity_id):
        return float(self.data.get("dhw_setpoint", 50.0))

    def get_hvac_mode(self, entity_id):
        return HvacMode(HvacMode.OFF)

    def get_forecast(self, entity_id):
        return self.data.get("solcast", {})

    def get_weather(self):
        return self.data.get("weather", {})