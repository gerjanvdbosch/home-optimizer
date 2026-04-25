from __future__ import annotations

from typing import Any, Protocol

from home_optimizer.domain.timeseries import MinuteSample


class LiveStateGateway(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...


class LiveSampleRepository(Protocol):
    @property
    def source(self) -> str: ...

    def write_samples(self, samples: list[MinuteSample]) -> None: ...

