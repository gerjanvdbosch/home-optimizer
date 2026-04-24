from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader(ABC):
    @abstractmethod
    def load(self) -> dict[str, Any]:
        pass


class AddonConfigLoader(ConfigLoader):
    def __init__(self, path: str = "/data/options.json") -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Addon options not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid addon options: {self.path}")

        return data


class LocalConfigLoader(ConfigLoader):
    def __init__(self, path: str = "config.yaml") -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Local config not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Invalid local config: {self.path}")

        options = data.get("options", data)

        if not isinstance(options, dict):
            raise ValueError(f"Invalid options section in: {self.path}")

        return options