import requests
import pandas as pd
import logging

from config import Config
from context import Context

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context

    def get_forecast(self):
        sensors = {
            "temperature_2m",
            "cloud_cover",
            "wind_speed_10m",
            "shortwave_radiation_instant",
            "diffuse_radiation_instant",
            "global_tilted_irradiance_instant",
        }

        if self.context.latitude is None or self.context.longitude is None:
            logger.error("[Weather] Geen locatiegegevens beschikbaar")
            return pd.DataFrame()

        params = {
            "latitude": self.context.latitude,
            "longitude": self.context.longitude,
            "tilt": self.config.pv_tilt,
            "azimuth": self.config.pv_azimuth,
            "minutely_15": ",".join(sensors),
            "timezone": "UTC",
            "forecast_days": 2,
            "past_days": 1,
        }

        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast", params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            minutely = data.get("minutely_15", {})

            if not minutely:
                logger.error("[Weather] Geen 15-min data ontvangen")
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(minutely["time"], utc=True),
                    "temp": minutely["temperature_2m"],
                    "cloud": minutely["cloud_cover"],
                    "wind": minutely["wind_speed_10m"],
                    "radiation": minutely["shortwave_radiation_instant"],
                    "diffuse": minutely["diffuse_radiation_instant"],
                    "tilted": minutely["global_tilted_irradiance_instant"],
                }
            )

            logger.debug(f"[Weather] API-update succesvol: {df}")
            return df

        except Exception as e:
            logger.error(f"[Weather] Fout bij ophalen OpenMeteo: {e}")
            return pd.DataFrame()
