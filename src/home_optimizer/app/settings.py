from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.sensors import SENSOR_CONFIG_KEYS
from home_optimizer.domain.types import JsonDict

DEFAULT_DATABASE_PATH = "/config/home_optimizer.db"


def _load_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")

    return data


def _load_yaml(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")

    options = data.get("options", data)
    if not isinstance(options, dict):
        raise ValueError(f"Invalid options section in: {path}")

    return options


class SensorBinding(DomainModel):
    entity_id: str

    @field_validator("entity_id", mode="before")
    @classmethod
    def _normalize_entity_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            raise ValueError("entity_id is required")
        return value


class AppSettings(DomainModel):
    database_path: str = DEFAULT_DATABASE_PATH
    api_port: int = Field(default=8099, ge=1, le=65535)
    history_import_enabled: bool = True
    history_import_chunk_days: int = Field(default=3, gt=0)
    history_import_max_days_back: int = Field(default=10, gt=0)
    pv_tilt: float | None = Field(default=None, ge=0, le=90)
    pv_azimuth: float | None = Field(default=None, ge=0, lt=360)
    boiler_tank_liters: int | None = Field(default=None, gt=0)
    sensors: dict[str, SensorBinding] = Field(default_factory=dict)

    sensor_room_temperature: str | None = None
    sensor_outdoor_temperature: str | None = None
    sensor_thermostat_setpoint: str | None = None
    sensor_shutter_living_room: str | None = None
    sensor_hp_supply_temperature: str | None = None
    sensor_hp_supply_target_temperature: str | None = None
    sensor_hp_return_temperature: str | None = None
    sensor_hp_flow: str | None = None
    sensor_hp_mode: str | None = None
    sensor_compressor_frequency: str | None = None
    sensor_defrost_active: str | None = None
    sensor_booster_heater_active: str | None = None
    sensor_refrigerant_condensation_temperature: str | None = None
    sensor_refrigerant_liquid_line_temperature: str | None = None
    sensor_discharge_temperature: str | None = None
    sensor_dhw_top_temperature: str | None = None
    sensor_dhw_bottom_temperature: str | None = None
    sensor_boiler_ambient_temperature: str | None = None
    sensor_hp_electric_power: str | None = None
    sensor_p1_net_power: str | None = None
    sensor_pv_output_power: str | None = None
    sensor_pv_total_kwh: str | None = None
    sensor_hp_electric_total_kwh: str | None = None
    sensor_p1_import_total_kwh: str | None = None
    sensor_p1_export_total_kwh: str | None = None

    @field_validator("history_import_max_days_back", mode="before")
    @classmethod
    def _default_empty_history_window(cls, value: Any) -> Any:
        return 10 if value in (None, "") else value

    @field_validator(*SENSOR_CONFIG_KEYS, mode="before")
    @classmethod
    def _normalize_optional_sensor(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return None if value in (None, "") else value

    @classmethod
    def from_addon_file(cls, path: str = "/data/options.json") -> "AppSettings":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Addon options not found: {config_path}")

        return cls.from_options(_load_json(config_path))

    @classmethod
    def from_local_file(cls, path: str = "config.yaml") -> "AppSettings":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Local config not found: {config_path}")

        return cls.from_options(_load_yaml(config_path))

    @classmethod
    def from_options(cls, options: JsonDict) -> "AppSettings":
        return cls.model_validate(options)
