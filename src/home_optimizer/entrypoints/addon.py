from __future__ import annotations

import logging

import uvicorn

from home_optimizer.app.container import build_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings import AppSettings
from home_optimizer.web import create_app

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    LOGGER.info("Starting Home Optimizer Add-on web API")

    settings = AppSettings.from_addon_file("/data/options.json")
    app = create_app(settings, container_factory=build_container)
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)


if __name__ == "__main__":
    main()
