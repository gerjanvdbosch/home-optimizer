from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from domain.models.config import DataSpec


class Storage(Protocol):
    def save(self, data: dict[str, Any]) -> None: ...

    def load(self) -> dict[str, Any]: ...


class DataLoader(Protocol):
    def supports(self, spec: DataSpec) -> bool: ...

    def load(self, spec: DataSpec, start: datetime, end: datetime) -> pd.DataFrame: ...
