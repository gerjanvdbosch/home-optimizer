from __future__ import annotations

import logging

from home_optimizer.bootstrap.dependencies import build_container
from home_optimizer.bootstrap.logging import configure_logging
from home_optimizer.bootstrap.settings import AppSettings
from home_optimizer.features.history_import.schemas import HistoryImportRequest
from home_optimizer.shared.sensors.factory import build_sensor_specs

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    LOGGER.info("Starting Home Optimizer locally")

    settings = AppSettings.from_local_file("config.yaml")
    container = build_container(settings)

    try:
        if settings.history_import_enabled:
            request = HistoryImportRequest.from_settings(
                settings=settings,
                specs=build_sensor_specs(settings.options or {}),
            )
            result = container.history_import_service.import_many(request)

            for name, count in result.imported_rows.items():
                LOGGER.info("%s: imported %s rows", name, count)
        else:
            LOGGER.info("Historical import disabled")
    finally:
        container.home_assistant.close()


if __name__ == "__main__":
    main()
