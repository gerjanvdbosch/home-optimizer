import os
import requests
import logging

from utils import safe_float, to_kw
from context import HvacMode
from config import Config

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self, config: Config):
        self.config = config
        self.url = os.environ.get("SUPERVISOR_API", "http://supervisor/core/api")
        self.token = os.environ.get("SUPERVISOR_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_location(self):
        attributes = self._get_attributes(self.config.sensor_home)
        logger.debug(f"[Client] Location attributes: {attributes}")

        if not attributes:
            return None, None

        latitude = safe_float(attributes.get("latitude"))
        longitude = safe_float(attributes.get("longitude"))
        return latitude, longitude

    def get_pv_power(self):
        return to_kw(self._get_state(self.config.sensor_pv_power))

    def get_wp_power(self):
        return to_kw(self._get_state(self.config.sensor_wp_power))

    def get_grid_power(self):
        return to_kw(self._get_state(self.config.sensor_grid_power))

    def get_room_temp(self):
        return safe_float(self._get_state(self.config.sensor_room_temp))

    def get_dhw_top(self):
        return safe_float(self._get_state(self.config.sensor_dhw_top))

    def get_dhw_bottom(self):
        return safe_float(self._get_state(self.config.sensor_dhw_bottom))

    def get_dhw_setpoint(self):
        return safe_float(self._get_state(self.config.sensor_dhw_setpoint))

    def get_supply_temp(self):
        return safe_float(self._get_state(self.config.sensor_supply_temp))

    def get_return_temp(self):
        return safe_float(self._get_state(self.config.sensor_return_temp))

    def get_hvac_mode(self):
        hvac = {
            "Uit": HvacMode.OFF,
            "SWW": HvacMode.DHW,
            "Verwarmen": HvacMode.HEATING,
            "Koelen": HvacMode.COOLING,
            "Legionellapreventie": HvacMode.LEGIONELLA_PREVENTION,
            "Vorstbescherming": HvacMode.FROST_PROTECTION,
        }.get(self._get_state(self.config.sensor_hvac), HvacMode.OFF)
        return HvacMode(hvac)

    def get_forecast(self):
        attributes_today = self._get_attributes(self.config.sensor_solcast_today)
        attributes_tomorrow = self._get_attributes(self.config.sensor_solcast_tomorrow)

        forecast_today = []
        forecast_tomorrow = []

        if attributes_today:
            forecast_today = attributes_today.get("detailedForecast", [])

        if attributes_tomorrow:
            forecast_tomorrow = attributes_tomorrow.get("detailedForecast", [])

        return forecast_today + forecast_tomorrow

    def _get_state(self, entity_id):
        try:
            return self._get_payload(entity_id).get("state")
        except Exception as e:
            logger.exception("[Client] Error getting state for %s: %s", entity_id, e)
            return None

    def _get_attributes(self, entity_id):
        try:
            return self._get_payload(entity_id).get("attributes", {})
        except Exception as e:
            logger.exception(
                "[Client] Error getting attributes for %s: %s", entity_id, e
            )
            return None

    def _get_payload(self, entity_id):
        try:
            r = requests.get(
                f"{self.url}/states/{entity_id}", headers=self.headers, timeout=10
            )
            r.raise_for_status()
            payload = r.json()
            logger.debug("[Client] Payload for %s: %s", entity_id, payload)
            return payload
        except Exception as e:
            logger.exception("[Client] Error getting state %s: %s", entity_id, e)
            return None

    def _set_state(self, entity_id, state, attributes=None, friendly_name=None):
        if attributes is None:
            attributes = {}

        if friendly_name:
            attributes["friendly_name"] = friendly_name

        url = f"{self.url}/states/{entity_id}"
        payload = {"state": state, "attributes": attributes}

        try:
            r = requests.post(url, json=payload, headers=self.headers)
            r.raise_for_status()
            logger.debug(
                f"[Client] State set for {entity_id}: {state} (attrs: {len(attributes)})"
            )
            return True
        except Exception as e:
            logger.error(
                f"[Client] Failed to set state for {entity_id} with payload {payload}: {e}"
            )
            return False

    def _call_service(self, domain, service, data):
        try:
            r = requests.post(
                f"{self.url}/services/{domain}/{service}",
                json=data,
                headers=self.headers,
                timeout=10,
            )
            r.raise_for_status()
            return r.json() if r.text else {}
        except Exception as e:
            logger.exception(
                "[Client] Error calling service %s.%s: %s", domain, service, e
            )
            return None
