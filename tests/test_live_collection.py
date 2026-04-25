from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from home_optimizer.app.live_collection_scheduler import LiveCollectionScheduler
from home_optimizer.domain.sensors import SensorDefinition, SensorSpec
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
        definition=SensorDefinition(
            name=name,
            category="building",
            unit="degC",
            method="mean",
            poll_interval_seconds=poll_interval_seconds,
        ),
        entity_id=f"sensor.{name}",
    )


def dt(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 4, 25, 12, minute, second, tzinfo=timezone.utc)


def test_live_collection_collects_sensor_into_minute_buffer() -> None:
    gateway = FakeLiveGateway()
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec(poll_interval_seconds=5)],
    )

    assert service.collect_sensor(live_spec(), dt(0, 0)) is True
    assert gateway.calls == 1
    assert service.buffer.has_samples() is True


def test_live_collection_flushes_complete_minutes_every_five_minutes() -> None:
    gateway = FakeLiveGateway("10")
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec(poll_interval_seconds=5)],
    )

    service.collect_sensor(live_spec(), dt(0, 1))
    gateway.state = "20"
    service.collect_sensor(live_spec(), dt(5, 1))
    assert service.flush_complete_minutes(dt(5, 1)) == 1

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

    service.collect_sensor(live_spec(), dt(0, 1))
    assert repository.writes == []

    assert service.flush_all() == 1
    sample = repository.writes[-1][0]
    assert sample.timestamp_minute == dt(0, 0)
    assert sample.source == "home_assistant_live"


def test_live_collection_scheduler_registers_poll_and_flush_jobs() -> None:
    service = LiveCollectionService(
        gateway=FakeLiveGateway(),
        repository=FakeLiveRepository(),
        specs=[
            live_spec("room_temperature", poll_interval_seconds=5),
            live_spec("outdoor_temperature", poll_interval_seconds=30),
        ],
    )
    scheduler = LiveCollectionScheduler(service)

    scheduler.start()
    try:
        jobs = {job.id: job for job in scheduler.scheduler.get_jobs()}
        assert set(jobs) == {
            "live:collect:room_temperature",
            "live:collect:outdoor_temperature",
            "live:flush",
        }
        assert jobs["live:collect:room_temperature"].trigger.interval.total_seconds() == 5
        assert jobs["live:collect:outdoor_temperature"].trigger.interval.total_seconds() == 30
        assert jobs["live:flush"].trigger.interval.total_seconds() == 300
    finally:
        scheduler.stop()


def test_live_collection_scheduler_flushes_buffer_on_stop() -> None:
    gateway = FakeLiveGateway()
    repository = FakeLiveRepository()
    service = LiveCollectionService(
        gateway=gateway,
        repository=repository,
        specs=[live_spec()],
    )
    scheduler = LiveCollectionScheduler(service)

    service.collect_sensor(live_spec(), dt(0, 1))

    scheduler.stop()

    assert repository.writes[-1][0].timestamp_minute == dt(0, 0)
