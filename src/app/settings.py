import json
import os
from pathlib import Path

from dotenv import load_dotenv
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


def load_settings() -> Settings:
    options = Path("/data/options.json")

    if options.exists():
        return Settings(**json.loads(options.read_text()))

    load_dotenv()

    return Settings(
        influx_host=os.getenv("INFLUX_HOST", "homeassistant.local"),
        influx_port=int(os.getenv("INFLUX_PORT", 8086)),
        influx_username=os.getenv("INFLUX_USERNAME", ""),
        influx_password=os.getenv("INFLUX_PASSWORD", ""),
        influx_database=os.getenv("INFLUX_DATABASE", "home_assistant"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
