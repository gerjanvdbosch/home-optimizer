import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from app.container import Container
from app.state_update import StateUpdater
from app.trainer import Trainer
from domain.models.config import Settings
from features.dataset.builder import DatasetLoader
from features.dataset.loader import ForecastLoader, TimeSeriesLoader
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.storage import JsonStorage


def create_container() -> Container:
    settings = load_settings()

    configure_logger(settings.log_level)

    influx = InfluxDatabase(settings)
    resolver = InfluxSensorResolver(influx)

    dataset_loader = DatasetLoader(
        loaders=[
            TimeSeriesLoader(influx, resolver),
            ForecastLoader(influx, resolver),
        ],
    )

    trainer = Trainer(
        forecasters=[],
        storage=JsonStorage(
            settings.data_path / "training.json",
            format=True,
        ),
    )

    state_updater = StateUpdater(
        loader=dataset_loader,
        storage=JsonStorage(
            settings.data_path / "state.json",
        ),
    )

    return Container(
        state_updater=state_updater,
        trainer=trainer,
    )


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


def configure_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
