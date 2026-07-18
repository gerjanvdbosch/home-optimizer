import json
import os
from pathlib import Path

from dotenv import load_dotenv

from domain.models import Settings


def load_settings() -> Settings:
    options = Path("/data/options.json")

    if options.exists():
        return Settings(
            **json.loads(options.read_text()),
            data_path=Path("/data"),
        )

    load_dotenv()

    return Settings(
        influx_host=os.getenv("INFLUX_HOST", "homeassistant.local"),
        influx_port=int(os.getenv("INFLUX_PORT", 8086)),
        influx_username=os.getenv("INFLUX_USERNAME", ""),
        influx_password=os.getenv("INFLUX_PASSWORD", ""),
        influx_database=os.getenv("INFLUX_DATABASE", "home_assistant"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
