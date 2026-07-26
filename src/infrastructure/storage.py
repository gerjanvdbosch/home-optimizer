import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonStorage:
    def __init__(self, path: str | Path, format: bool = False):
        self.path = Path(path)
        self.format = format

    def save(self, obj: dict[str, Any] | BaseModel) -> None:
        if isinstance(obj, BaseModel):
            obj = obj.model_dump(mode="json")

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(
                obj,
                default=str,
                indent=2 if self.format else None,
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


# class JoblibStorage
