"""Machine-learning forecast providers for optimizer request enrichment."""

from .baseload import BaseloadForecaster
from .dhw_tap import DHWTapForecaster
from .models import BaseloadForecastSettings, DHWTapForecastSettings, ForecastServiceSettings, ShutterForecastSettings
from .service import BaseloadForecastProvider, DHWTapForecastProvider, ForecastProvider, ForecastService, ShutterForecastProvider
from .shutter import ShutterForecaster

__all__ = [
    "BaseloadForecaster",
    "BaseloadForecastProvider",
    "BaseloadForecastSettings",
    "DHWTapForecaster",
    "DHWTapForecastProvider",
    "DHWTapForecastSettings",
    "ForecastProvider",
    "ForecastService",
    "ForecastServiceSettings",
    "ShutterForecaster",
    "ShutterForecastProvider",
    "ShutterForecastSettings",
]
