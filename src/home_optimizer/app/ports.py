from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class SensorGateway(Protocol):
    def close(self) -> None: ...

    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def get_states(self) -> list[dict[str, Any]]: ...

    def get_location(self) -> tuple[float, float] | None: ...

    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        minimal_response: bool = True,
    ) -> list[dict[str, Any]]: ...
