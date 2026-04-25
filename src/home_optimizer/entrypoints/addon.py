from __future__ import annotations

import logging

import uvicorn

from home_optimizer.app.container_factories import build_home_assistant_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.web import create_app

LOGGER = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings("/data/options.json")
    configure_logging(settings.log_level)
    LOGGER.info("Starting Home Optimizer Add-on web API")

    app = create_app(settings, container_factory=build_home_assistant_container)
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)


if __name__ == "__main__":
    main()
