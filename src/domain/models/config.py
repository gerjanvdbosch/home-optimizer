from pydantic import BaseModel

from domain.models import SensorReference


class SolarConfig(BaseModel):
    production: SensorReference
    forecast: dict[str, SensorReference]


class BoilerTemperatureConfig(BaseModel):
    top: SensorReference
    bottom: SensorReference


class HeatPumpConfig(BaseModel):
    supply_temperature: SensorReference
    return_temperature: SensorReference
    boiler_temperature: BoilerTemperatureConfig
