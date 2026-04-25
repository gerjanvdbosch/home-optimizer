from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.types import JsonDict


def load_options_file(path: str) -> JsonDict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    options = data.get("options", data)
    if not isinstance(options, dict):
        raise ValueError(f"Invalid options section in: {config_path}")

    return options


def load_settings(path: str, overrides: list[str] | None = None) -> AppSettings:
    options = load_options_file(path)
    override_options = parse_dot_overrides(overrides or [])
    return AppSettings.from_options(deep_merge(options, override_options))


def parse_dot_overrides(overrides: list[str]) -> JsonDict:
    parsed: JsonDict = {}

    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid override: {override!r}. Expected key=value.")

        cursor: dict[str, Any] = parsed
        parts = key.split(".")
        for part in parts[:-1]:
            next_value = cursor.setdefault(part, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"Override path conflicts with scalar value: {key}")
            cursor = next_value

        cursor[parts[-1]] = _parse_override_value(raw_value)

    return parsed


def deep_merge(base: JsonDict, overrides: JsonDict) -> JsonDict:
    merged = dict(base)

    for key, override_value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value

    return merged


def _parse_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    return raw_value if value is None and raw_value else value

