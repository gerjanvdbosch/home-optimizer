# src/addon.py

from __future__ import annotations

import logging
from datetime import datetime, timezone

from client.homeassistant import HomeAssistantClient
from config.config_loaders import AddonConfigLoader
from config.sensor_definitions import build_sensor_specs
from database.session import Database
from importer.history_importer import (
    HomeAssistantHistoryImporter,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_HISTORY_START = "2026-04-14T00:00:00+02:00"


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        raise ValueError(f"Datetime must include a timezone: {value}")

    return dt


def run_initial_history_import(
    options: dict,
    db: Database,
) -> None:
    """
    Eenmalige historische import.

    Door chunk-skip logic veilig meerdere keren uitvoerbaar.
    """

    ha = HomeAssistantClient()

    try:
        specs = build_sensor_specs(options)

        importer = HomeAssistantHistoryImporter(
            ha_client=ha,
            database=db,
            chunk_days=int(options.get("history_import_chunk_days", 3)),
        )

        start_time = parse_datetime(
            str(options.get("history_import_start", DEFAULT_HISTORY_START))
        )
        end_option = options.get("history_import_end")
        end_time = parse_datetime(str(end_option)) if end_option else datetime.now(timezone.utc)

        result = importer.import_many(
            specs=specs,
            start_time=start_time,
            end_time=end_time,
        )

        for name, count in result.items():
            LOGGER.info("%s: imported %s rows", name, count)

    finally:
        ha.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Starting Home Optimizer Add-on")

    #
    # Settings
    #
    options = AddonConfigLoader("/data/options.json").load()

    #
    # Database
    #
    db = Database(str(options.get("database_path", "/data/home_optimizer.db")))
    db.init_schema()

    #
    # One-time historical import
    #
    if bool(options.get("history_import_enabled", True)):
        run_initial_history_import(
            options=options,
            db=db,
        )
    else:
        LOGGER.info("Historical import disabled")

    #
    # Daarna pas live collectors
    #
    LOGGER.info("Starting live collectors...")
    # start_collector_loop()
    # start_controller_loop()


if __name__ == "__main__":
    main()
