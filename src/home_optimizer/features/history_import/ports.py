from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.timeseries import MinuteSample


class HistorySourceGateway(Protocol):
    def get_history(
        self,
        *,
        entity_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        minimal_response: bool = True,
    ) -> list[dict[str, Any]]: ...


class HistoryRepository(Protocol):
    @property
    def source(self) -> str: ...

    def chunk_already_imported(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> bool: ...

    def mark_chunk_imported(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
        row_count: int,
    ) -> None: ...

    def write_samples(self, samples: list[MinuteSample]) -> None: ...

    def last_stored_value_before(
        self,
        spec: SensorSpec,
        before_time: datetime,
    ) -> Any | None: ...
