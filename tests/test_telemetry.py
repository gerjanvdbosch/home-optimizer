from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from home_optimizer.app.forecast_scheduler import ForecastScheduler
from home_optimizer.app.historical_weather_scheduler import HistoricalWeatherScheduler
from home_optimizer.app.telemetry_scheduler import TelemetryScheduler
from home_optimizer.domain.sensors import SensorDefinition, SensorSpec
from home_optimizer.domain.timeseries import MinuteSample
from home_optimizer.features.telemetry.service import TelemetryService


class FakeTelemetryGateway:
    def __init__(self, state: str = "10") -> None:
        self.state = state
        self.calls = 0

    def get_state(self, entity_id: str) -> dict[str, Any]:
        self.calls += 1
        return {"entity_id": entity_id, "state": self.state}


class FakeTelemetryRepository:
    source = "home_assistant_telemetry"

    def __init__(self) -> None:
        self.writes: list[list[MinuteSample]] = []

    def write_samples(self, samples: list[MinuteSample]) -> None:
        self.writes.append(samples)


class FakeForecastRunner:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls = 0

    def refresh_forecast(self) -> int:
        self.calls += 1
        return 1


class FakeHistoricalWeatherRunner:
    def __init__(self) -> None:
        self.calls = 0

    def import_historical_weather(self) -> int:
        self.calls += 1
        return 1


def telemetry_spec(name: str = "room_temperature", poll_interval_seconds: int = 5) -> SensorSpec:
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


def test_telemetry_collects_sensor_into_minute_buffer() -> None:
    gateway = FakeTelemetryGateway()
    repository = FakeTelemetryRepository()
    service = TelemetryService(
        gateway=gateway,
        repository=repository,
        specs=[telemetry_spec(poll_interval_seconds=5)],
    )

    assert service.collect_sensor(telemetry_spec(), dt(0, 0)) is True
    assert gateway.calls == 1
    assert service.buffer.has_samples() is True


def test_telemetry_flushes_complete_minutes_every_five_minutes() -> None:
    gateway = FakeTelemetryGateway("10")
    repository = FakeTelemetryRepository()
    service = TelemetryService(
        gateway=gateway,
        repository=repository,
        specs=[telemetry_spec(poll_interval_seconds=5)],
    )

    service.collect_sensor(telemetry_spec(), dt(0, 1))
    gateway.state = "20"
    service.collect_sensor(telemetry_spec(), dt(5, 1))

    assert service.flush_complete_minutes(dt(5, 1)) == 1

    written_batches = [batch for batch in repository.writes if batch]
    assert len(written_batches) == 1
    assert len(written_batches[0]) == 1
    sample = written_batches[0][0]
    assert sample.timestamp_minute == dt(0, 0)
    assert sample.mean_real == 10.0
    assert sample.last_real == 10.0
    assert sample.sample_count == 1


def test_telemetry_shutdown_flushes_current_minute() -> None:
    gateway = FakeTelemetryGateway("10")
    repository = FakeTelemetryRepository()
    service = TelemetryService(
        gateway=gateway,
        repository=repository,
        specs=[telemetry_spec()],
    )

    service.collect_sensor(telemetry_spec(), dt(0, 1))
    assert repository.writes == []

    assert service.flush_all() == 1
    sample = repository.writes[-1][0]
    assert sample.timestamp_minute == dt(0, 0)
    assert sample.source == "home_assistant_telemetry"


def test_telemetry_scheduler_registers_poll_and_flush_jobs() -> None:
    service = TelemetryService(
        gateway=FakeTelemetryGateway(),
        repository=FakeTelemetryRepository(),
        specs=[
            telemetry_spec("room_temperature", poll_interval_seconds=5),
            telemetry_spec("outdoor_temperature", poll_interval_seconds=30),
        ],
    )
    scheduler = TelemetryScheduler(service)

    scheduler.start()
    try:
        jobs = {job.id: job for job in scheduler.scheduler.get_jobs()}
        assert set(jobs) == {
            "telemetry:collect:room_temperature",
            "telemetry:collect:outdoor_temperature",
            "telemetry:flush",
        }
        assert jobs["telemetry:collect:room_temperature"].trigger.interval.total_seconds() == 5
        assert jobs["telemetry:collect:outdoor_temperature"].trigger.interval.total_seconds() == 30
        assert jobs["telemetry:flush"].trigger.interval.total_seconds() == 300
    finally:
        scheduler.stop()


def test_telemetry_scheduler_flushes_buffer_on_stop() -> None:
    gateway = FakeTelemetryGateway()
    repository = FakeTelemetryRepository()
    service = TelemetryService(
        gateway=gateway,
        repository=repository,
        specs=[telemetry_spec()],
    )
    scheduler = TelemetryScheduler(service)

    service.collect_sensor(telemetry_spec(), dt(0, 1))
    scheduler.stop()

    assert repository.writes[-1][0].timestamp_minute == dt(0, 0)


def test_forecast_scheduler_runs_once_on_start_and_registers_interval_job() -> None:
    runner = FakeForecastRunner()
    scheduler = ForecastScheduler(runner, interval_seconds=1800)

    scheduler.start()
    try:
        jobs = {job.id: job for job in scheduler.scheduler.get_jobs()}
        assert runner.calls == 1
        assert set(jobs) == {"forecast:refresh"}
        assert jobs["forecast:refresh"].trigger.interval.total_seconds() == 1800
    finally:
        scheduler.stop()


def test_forecast_scheduler_skips_start_when_disabled() -> None:
    runner = FakeForecastRunner(enabled=False)
    scheduler = ForecastScheduler(runner)

    scheduler.start()

    assert runner.calls == 0
    assert scheduler.scheduler.running is False


def test_historical_weather_scheduler_registers_daily_cron_job_at_1am() -> None:
    runner = FakeHistoricalWeatherRunner()
    scheduler = HistoricalWeatherScheduler(runner)

    scheduler.start()
    try:
        jobs = {job.id: job for job in scheduler.scheduler.get_jobs()}
        assert runner.calls == 0
        assert set(jobs) == {"historical-weather:import"}
        assert str(jobs["historical-weather:import"].trigger) == "cron[hour='1', minute='0']"
    finally:
        scheduler.stop()

