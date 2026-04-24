# src/addon.py

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from client.homeassistant import HomeAssistantClient
from config.sensor_factory import build_sensor_specs
from database.session import Database
from importer.history_importer import (
    HomeAssistantHistoryImporter,
)
from home_optimizer.settings import Settings


def run_initial_history_import(
    settings: Settings,
    db: Database,
) -> None:
    """
    Eenmalige historische import.

    Door chunk-skip logic veilig meerdere keren uitvoerbaar.
    """

    ha = HomeAssistantClient()

    try:
        specs = build_sensor_specs(settings)

        importer = HomeAssistantHistoryImporter(
            ha_client=ha,
            database=db,
            chunk_days=3,
        )

        start_time = datetime(
            2026,
            4,
            14,
            tzinfo=ZoneInfo("Europe/Amsterdam"),
        )

        result = importer.import_many(
            specs=specs,
            start_time=start_time,
        )

        for name, count in result.items():
            print(f"{name}: imported {count} rows")

    finally:
        ha.close()


def main() -> None:
    print("Starting Home Optimizer Add-on")

    #
    # Settings
    #
    settings = Settings("/data/options.json")

    #
    # Database
    #
    db = Database("/data/home_optimizer.db")
    db.init_schema()

    #
    # One-time historical import
    #
    run_initial_history_import(
        settings=settings,
        db=db,
    )

    #
    # Daarna pas live collectors
    #
    print("Starting live collectors...")
    # start_collector_loop()
    # start_controller_loop()


if __name__ == "__main__":
    main()