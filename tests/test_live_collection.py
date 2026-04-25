from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.timeseries import MinuteSample
from home_optimizer.features.live_collection.service import LiveCollectionService


class FakeLiveGateway:
    def __init__(self, state: str = "10") -> None:
        self.state = state
        self.calls = 0

    def get_state(self, entity_id: str) -> dict[str, Any]:
        self.calls += 1
        return {"entity_id": entity_id, "state": self.state}


class FakeLiveRepository:
    source = "home_assistant_live"

    def __init__(self) -> None:
        self.writes: list[list[MinuteSample]] = []

    def write_samples(self, samples: list[MinuteSample]) -> None:
        self.writes.append(samples)


def live_spec(name: str = "room_temperature", poll_interval_seconds: int = 5) -> SensorSpec:
    return SensorSpec(
        name=name,
        entity_id=f"sensor.{name}",
        category="building",
        unit="degC",
        method="mean",
        poll_interval_seconds=poll_interval_seconds,
    )


def dt(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 4, 25, 12, minute, second, tzinfo=timezone.utc)


def test_live_collection_respects_poll_interval() -> None:
    gateway = FakeLiveGateway()
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec(poll_interval_seconds=5)],
    )

    assert service.collect_due(dt(0, 0)) == 1
    assert service.collect_due(dt(0, 4)) == 0
    assert service.collect_due(dt(0, 5)) == 1
    assert gateway.calls == 2


def test_live_collection_flushes_complete_minutes_every_five_minutes() -> None:
    gateway = FakeLiveGateway("10")
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec(poll_interval_seconds=5)],
        flush_interval_seconds=300,
    )

    service.collect_due(dt(0, 1))
    gateway.state = "20"
    service.collect_due(dt(5, 1))

    written_batches = [batch for batch in repository.writes if batch]
    assert len(written_batches) == 1
    assert len(written_batches[0]) == 1
    sample = written_batches[0][0]
    assert sample.timestamp_minute == dt(0, 0)
    assert sample.mean_real == 10.0
    assert sample.last_real == 10.0
    assert sample.sample_count == 1


def test_live_collection_shutdown_flushes_current_minute() -> None:
    gateway = FakeLiveGateway("10")
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec()],
    )

    service.collect_due(dt(0, 1))
    assert repository.writes == []

    assert service.flush(include_current_minute=True) == 1
    sample = repository.writes[-1][0]
    assert sample.timestamp_minute == dt(0, 0)
    assert sample.source == "home_assistant_live"

