import os

from dataclasses import dataclass
from utils import safe_float


@dataclass
class Config:
    pv_azimuth: float = 148.0
    pv_tilt: float = 50.0
    pv_max_kw: float = 2.0

    sensor_pv_power: str = "sensor.pv_output"
    sensor_wp_power: str = "sensor.warmtepomp_geschat_vermogen"
    sensor_grid_power: str = "sensor.p1_meter_power"

    sensor_pv_energy = "sensor.pv_energie_totaal"
    sensor_wp_energy = "sensor.warmtepomp_verbruikt_gerapporteerd_totaal"
    sensor_grid_import_energy = "sensor.p1_meter_energy_import"
    sensor_grid_export_energy = "sensor.p1_meter_energy_export"

    sensor_dhw_temp: str = "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
    sensor_dhw_setpoint: str = "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde"
    sensor_hvac: str = "sensor.ecodan_heatpump_ca09ec_status_bedrijf"

    sensor_solcast: str = "sensor.solcast_pv_forecast_forecast_today"
    sensor_home: str = "zone.home"

    database_path: str = "/config/db/database.sqlite"

    load_model_path: str = "/config/models/load_model.joblib"

    solar_model_path: str = "/config/models/solar_model.joblib"
    solar_model_ratio: float = 0.0

    webapi_host: str = "0.0.0.0"
    webapi_port: int = 8000

    @staticmethod
    def load():
        return Config(solar_model_ratio=safe_float(os.getenv("SOLAR_MODEL_RATIO", 0.7)))
