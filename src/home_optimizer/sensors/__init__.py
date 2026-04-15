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
        room_temperature=HAEntityConfig("sensor.living_room_temperature"),
        outdoor_temperature=HAEntityConfig("sensor.outdoor_temperature"),
        hp_supply_temperature=HAEntityConfig("sensor.heat_pump_supply_temperature"),
        hp_return_temperature=HAEntityConfig("sensor.heat_pump_return_temperature"),
        hp_flow_lpm=HAEntityConfig("sensor.heat_pump_flow_lpm"),
        hp_electric_power=HAEntityConfig("sensor.heat_pump_power_kw"),
        hp_mode_entity_id="sensor.heat_pump_mode",
        p1_net_power=HAEntityConfig("sensor.p1_net_power_w", scale=0.001),
        pv_output=HAEntityConfig("sensor.pv_inverter_power_w", scale=0.001),
        thermostat_setpoint=HAEntityConfig("sensor.room_setpoint_temperature"),
        dhw_top_temperature=HAEntityConfig("sensor.dhw_top_temperature"),
        dhw_bottom_temperature=HAEntityConfig("sensor.dhw_bottom_temperature"),
        shutter_living_room=HAEntityConfig("sensor.shutter_living_room_pct"),
        defrost_active_entity_id="binary_sensor.hp_defrost",
        booster_heater_active_entity_id="binary_sensor.dhw_booster_heater",
        boiler_ambient_temperature=HAEntityConfig("sensor.boiler_ambient_temp"),
        refrigerant_condensation_temperature=HAEntityConfig("sensor.refrigerant_cond_temp"),
        refrigerant_temperature=HAEntityConfig("sensor.refrigerant_evap_temp"),
        base_url="http://homeassistant.local:8123",   # omit in addon mode
        token="YOUR_LONG_LIVED_TOKEN",                # omit in addon mode
    )
    readings = backend.read_all()
    print(readings.room_temperature_c, readings.hp_electric_power_kw, readings.hp_mode)

    # 2. Weather forecast from Open-Meteo (Amsterdam, south-facing windows)
    weather = OpenMeteoClient(latitude=52.37, longitude=4.90).get_forecast(horizon_hours=24)

    # 3. Combine into ForecastHorizon for the MPC
    forecast = build_forecast(
        weather,
        price_eur_per_kwh=0.25,
        pv_power_kw=readings.pv_output_kw,
        room_temperature_ref_c=21.0,
    )

Quick-start (local / standalone, no HA)
----------------------------------------
::

    from home_optimizer.sensors import LocalBackend, OpenMeteoClient, build_forecast

    backend = LocalBackend.from_json_file("sensors.json")
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

