from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATABASE_PATH = "/config/home_optimizer.db"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")

    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")

    options = data.get("options", data)
    if not isinstance(options, dict):
        raise ValueError(f"Invalid options section in: {path}")

    return options


@dataclass(frozen=True)
class AppSettings:
    database_path: str = DEFAULT_DATABASE_PATH
    history_import_enabled: bool = True
    history_import_chunk_days: int = 3
    history_import_max_days_back: int = 10
    options: dict[str, Any] | None = None

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
    def from_options(cls, options: dict[str, Any]) -> "AppSettings":
        history_import_max_days_back = options.get("history_import_max_days_back")

        return cls(
            database_path=str(options.get("database_path", DEFAULT_DATABASE_PATH)),
            history_import_enabled=bool(options.get("history_import_enabled", True)),
            history_import_chunk_days=int(options.get("history_import_chunk_days", 3)),
            history_import_max_days_back=int(
                history_import_max_days_back
                if history_import_max_days_back not in (None, "")
                else 10
            ),
            options=options,
        )
