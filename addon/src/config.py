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

    sensor_dhw_top: str = "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor"
    sensor_dhw_bottom: str = "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
    sensor_dhw_setpoint: str = "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde"

    sensor_room_temp: str = "sensor.danfoss_15_temperature"
    sensor_compressor_freq: str = "sensor.ecodan_heatpump_compressor_frequentie"
    sensor_supply_temp: str = "sensor.ecodan_heatpump_ca09ec_aanvoer_temp"
    sensor_hvac: str = "sensor.ecodan_heatpump_ca09ec_status_bedrijf"

    sensor_solcast: str = "sensor.solcast_pv_forecast_forecast_today"
    sensor_home: str = "zone.home"

    database_path: str = "/config/db/database.sqlite"

    ufh_model_path: str = "/config/models/ufh_model.joblib"
    dhw_model_path: str = "/config/models/dhw_model.joblib"
    rc_model_path: str = "/config/models/rc_model.joblib"
    load_model_path: str = "/config/models/load_model.joblib"
    solar_model_path: str = "/config/models/solar_model.joblib"

    solar_model_ratio: float = 0.0

    webapi_host: str = "0.0.0.0"
    webapi_port: int = 8000

    @staticmethod
    def load():
        return Config(solar_model_ratio=safe_float(os.getenv("SOLAR_MODEL_RATIO", 0.7)))
