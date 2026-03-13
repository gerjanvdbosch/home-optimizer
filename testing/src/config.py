import os

from dataclasses import dataclass
from utils import safe_float


@dataclass
class Config:
    avg_price: float = 0.22
    export_price: float = 0.07

    pv_azimuth: float = 148.0
    pv_tilt: float = 50.0
    pv_max_kw: float = 2.0

    tank_liters: int = 200

    sensor_pv_power: str = "sensor.pv_output"
    sensor_grid_power: str = "sensor.p1_meter_power"
    sensor_wp_power: str = "sensor.warmtepomp_geschat_vermogen"

    sensor_dhw_top: str = "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor"
    sensor_dhw_bottom: str = "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
    sensor_room_temp: str = "sensor.danfoss_15_temperature"
    sensor_target_setpoint: str = "sensor.warmtepomp_setpoint_waarde"
    sensor_supply_temp: str = "sensor.ecodan_heatpump_ca09ec_aanvoer_temp"
    sensor_return_temp: str = "sensor.ecodan_heatpump_ca09ec_retour_temp"
    sensor_hvac: str = "sensor.ecodan_heatpump_ca09ec_status_bedrijf"

    sensor_shutter_room: str = "sensor.woonkamer_rolluik"

    sensor_solcast_today: str = "sensor.solcast_pv_forecast_forecast_today"
    sensor_solcast_tomorrow: str = "sensor.solcast_pv_forecast_forecast_tomorrow"
    sensor_home: str = "zone.home"

    database_path: str = "data/database.sqlite"

    ufh_model_path: str = "data/ufh_model.joblib"
    dhw_model_path: str = "data/dhw_model.joblib"
    rc_model_path: str = "data/rc_model.joblib"
    load_model_path: str = "data/load_model.joblib"
    solar_model_path: str = "data/solar_model.joblib"
    hp_model_path: str = "data/hp_model.joblib"
    hydraulic_model_path: str = "data/hydraulic_model.joblib"
    shutter_model_path: str = "data/shutter_model.joblib"

    solar_model_ratio: float = 0.0

    webapi_host: str = "127.0.0.1"
    webapi_port: int = 8000

    @staticmethod
    def load():
        return Config(solar_model_ratio=safe_float(os.getenv("SOLAR_MODEL_RATIO", 0.7)))
