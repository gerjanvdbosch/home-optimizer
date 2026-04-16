"""Minimal telemetry collection demo.

Run this example to persist aggregated 5-minute telemetry buckets to SQLite while
sampling the local JSON sensor file every 30 seconds.
"""

from __future__ import annotations

import time
from pathlib import Path

from home_optimizer.sensors import LocalBackend
from home_optimizer.telemetry import (
    BufferedTelemetryCollector,
    TelemetryCollectorSettings,
    TelemetryRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSORS_JSON = PROJECT_ROOT / "sensors.json"
TELEMETRY_SQLITE = PROJECT_ROOT / "database.sqlite3"
RUN_DURATION_SECONDS: int = 20


def main() -> None:
    settings = TelemetryCollectorSettings(
        database_url=f"sqlite:///{TELEMETRY_SQLITE}",
    )
    repository = TelemetryRepository(database_url=settings.database_url)
    backend = LocalBackend.from_json_file(SENSORS_JSON)
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=settings,
    )

    collector.start()
    print(f"Collecting telemetry into {TELEMETRY_SQLITE} for {RUN_DURATION_SECONDS} seconds...")
    try:
        time.sleep(RUN_DURATION_SECONDS)
    finally:
        collector.shutdown(flush=True)
        print(f"Persisted {repository.count()} aggregated bucket(s).")


if __name__ == "__main__":
    main()
