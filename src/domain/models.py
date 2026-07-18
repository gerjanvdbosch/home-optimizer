from pydantic import BaseModel, Field


class Settings(BaseModel):
    influx_host: str = Field(
        default="homeassistant.local",
        description="InfluxDB host",
    )
    influx_port: int = Field(
        default=8086,
        description="InfluxDB port",
    )
    influx_username: str = Field(
        default="",
        description="InfluxDB username",
    )
    influx_password: str = Field(
        default="",
        description="InfluxDB password",
    )
    influx_database: str = Field(
        default="home_assistant",
        description="InfluxDB database",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


class SolarForecast(BaseModel):
    p10: tuple[str, str]
    p50: tuple[str, str]
    p90: tuple[str, str]


class UpdateRequest(BaseModel):
    solar_forecast: SolarForecast
