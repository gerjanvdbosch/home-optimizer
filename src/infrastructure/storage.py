import json
from pathlib import Path
from typing import Any

from domain.models import Storage


class JsonStorage(Storage):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(
                data,
                default=str,
            ),
            encoding="utf-8",
        )

        temp_path.replace(self.path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        return json.loads(
            self.path.read_text(
                encoding="utf-8",
            )
        )


class MemoryStorage(Storage):
    def __init__(self):
        self.data: dict[str, Any] = {}

    def save(self, data: dict[str, Any]) -> None:
        self.data = data

    def load(self) -> dict[str, Any]:
        return self.data
