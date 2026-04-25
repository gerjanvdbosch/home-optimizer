from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.types import JsonDict

DEFAULT_DATABASE_PATH = "/config/home_optimizer.db"


class AppSettings(DomainModel):
    model_config = ConfigDict(extra="forbid")

    database_path: str = DEFAULT_DATABASE_PATH
    api_port: int = Field(default=8099, ge=1, le=65535)
    history_import_enabled: bool = True
    history_import_chunk_days: int = Field(default=3, gt=0)
    history_import_max_days_back: int = Field(default=10, gt=0)
    pv_tilt: float | None = Field(default=None, ge=0, le=90)
    pv_azimuth: float | None = Field(default=None, ge=0, lt=360)
    boiler_tank_liters: int | None = Field(default=None, gt=0)
    sensors: dict[str, str] = Field(default_factory=dict)

    @field_validator("sensors", mode="before")
    @classmethod
    def _normalize_sensors(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value

        sensors: dict[str, str] = {}
        for name, entity_id in value.items():
            if isinstance(entity_id, str):
                entity_id = entity_id.strip()
            if entity_id in (None, ""):
                continue
            sensors[str(name)] = entity_id

        return sensors

    @field_validator("history_import_max_days_back", mode="before")
    @classmethod
    def _default_empty_history_window(cls, value: Any) -> Any:
        return 10 if value in (None, "") else value

    @classmethod
    def from_options(cls, options: JsonDict) -> "AppSettings":
        return cls.model_validate(options)
