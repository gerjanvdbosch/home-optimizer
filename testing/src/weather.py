import pandas as pd
import logging

from client import HAClient
from context import Context

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, client: HAClient, context: Context):
        self.client = client

    def get_forecast(self):
        minutely = self.client.get_weather()

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

        logger.debug("[Weather] API-update succesvol")
        return df
