from datetime import datetime
from typing import Any, Protocol, TypeVar

import pandas as pd

from domain.models.config import DataSpec


class Storage(Protocol):
    def save(self, data: dict[str, Any]) -> None: ...

    def load(self) -> dict[str, Any]: ...


T = TypeVar("T", bound=DataSpec, contravariant=True)


class DataLoader(Protocol[T]):
    def supports(self, spec: DataSpec) -> bool: ...

    def load(self, spec: T, start: datetime, end: datetime) -> pd.DataFrame: ...
