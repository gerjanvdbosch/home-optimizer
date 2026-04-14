"""Sensor backends and weather forecast client for Home Optimizer.

Quick-start (Home Assistant addon / standalone)
------------------------------------------------
::

    from home_optimizer.sensors import (
        HomeAssistantBackend, HAEntityConfig,
        OpenMeteoClient,
        build_forecast,
    )

    # 1. Live sensor readings from HA
    backend = HomeAssistantBackend(
        room_temp=HAEntityConfig("sensor.living_room_temperature"),
        pv_power=HAEntityConfig("sensor.pv_inverter_power_w", scale=0.001),  # W -> kW
        hp_power=HAEntityConfig("sensor.heat_pump_power_kw"),
        base_url="http://homeassistant.local:8123",   # omit in addon mode
        token="YOUR_LONG_LIVED_TOKEN",                # omit in addon mode
    )
    readings = backend.read_all()
    print(readings.room_temperature_c, readings.pv_power_kw, readings.hp_power_kw)

    # 2. Weather forecast from Open-Meteo (Amsterdam, south-facing windows)
    weather = OpenMeteoClient(latitude=52.37, longitude=4.90).get_forecast(horizon_hours=24)

    # 3. Combine into ForecastHorizon for the MPC
    forecast = build_forecast(
        weather,
        price_eur_per_kwh=0.25,
        pv_power_kw=readings.pv_power_kw,
        room_temperature_ref_c=21.0,
    )

Quick-start (local / standalone, no HA)
----------------------------------------
::

    from home_optimizer.sensors import LocalBackend, OpenMeteoClient, build_forecast

    backend = LocalBackend(room_temperature_c=20.5, pv_power_kw=1.2, hp_power_kw=2.0)
    # or read from env vars:  backend = LocalBackend.from_env()
"""

from .base import LiveReadings, SensorBackend
from .factory import build_forecast, effective_price
from .ha_backend import HAEntityConfig, HomeAssistantBackend
from .local_backend import LocalBackend
from .open_meteo import OpenMeteoClient, WeatherForecast

__all__ = [
    # Core abstractions
    "LiveReadings",
    "SensorBackend",
    # Home Assistant backend
    "HAEntityConfig",
    "HomeAssistantBackend",
    # Local / standalone backend
    "LocalBackend",
    # Weather forecast
    "OpenMeteoClient",
    "WeatherForecast",
    # ForecastHorizon factory
    "build_forecast",
    "effective_price",
]

