import json
import os

from client import HAClient
from dataclasses import dataclass
from utils import safe_float

@dataclass
class Config:
    latitude: float = 52.0
    longitude: float = 5.0

    pv_azimuth: float = 148.0
    pv_tilt: float = 50.0
    pv_max_kw: float = 2.0

    sensor_pv_power: str = "sensor.pv_output"
    sensor_grid_power: str = "sensor.p1_meter_power"
    sensor_wp_power: str = "sensor.warmtepomp_geschat_vermogen"

    sensor_pv_energy = "sensor.pv_energie_totaal"
    sensor_wp_energy = "sensor.warmtepomp_verbruikt_gerapporteerd_totaal"
    sensor_grid_import_energy = "sensor.p1_meter_energy_import"
    sensor_grid_export_energy = "sensor.p1_meter_energy_export"

    sensor_dhw_temp: str = "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
    sensor_dhw_setpoint: str = "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde"
    sensor_hvac: str = "sensor.ecodan_heatpump_ca09ec_status_bedrijf"

    sensor_solcast: str = "sensor.solcast_pv_forecast_forecast_today"
    sensor_home: str = "zone.home"

    database_path: str = "data/database.sqlite"

    load_model_path: str = "data/load_model.joblib"

    solar_model_path: str = "data/solar_model.joblib"
    solar_model_ratio: float = 0

    webapi_host: str = "127.0.0.1"
    webapi_port: int = 8000

    @staticmethod
    def load(client: HAClient):
        config = Config()

        location = client.get_location(config.sensor_home)
        if location != (None, None):
            config.latitude, config.longitude = location

        config.solar_model_ratio = safe_float(os.getenv("SOLAR_MODEL_RATIO", 0.7))

        return config
