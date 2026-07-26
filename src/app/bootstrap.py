import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.state import StateManager
from app.training import Trainer
from domain.mapper import StateMapper
from domain.models.config import Settings
from features.dataset import (
    AttributeSeriesLoader,
    AttributeTimeSeriesLoader,
    DatasetLoader,
    TimeSeriesLoader,
)
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.repository import ConfigRepository, StateRepository
from infrastructure.storage import JsonStorage


@dataclass(slots=True)
class Container:
    config_repository: ConfigRepository
    state_manager: StateManager
    trainer: Trainer


def create_container() -> Container:
    settings = load_settings()

    configure_logger(settings.log_level)

    influx = InfluxDatabase(settings)
    resolver = InfluxSensorResolver(influx)

    dataset_loader = DatasetLoader(
        loaders=[
            TimeSeriesLoader(influx, resolver),
            AttributeTimeSeriesLoader(influx, resolver),
            AttributeSeriesLoader(influx, resolver),
        ],
    )

    config_repository = ConfigRepository(
        JsonStorage(settings.data_path / "config.json"),
    )

    state_repository = StateRepository(
        JsonStorage(
            settings.data_path / "state.json",
        )
    )

    state_manager = StateManager(
        loader=dataset_loader,
        repository=state_repository,
        mapper=StateMapper(),
    )

    trainer = Trainer(
        loader=dataset_loader,
        forecasters=[],
    )

    return Container(
        config_repository=config_repository,
        state_manager=state_manager,
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
