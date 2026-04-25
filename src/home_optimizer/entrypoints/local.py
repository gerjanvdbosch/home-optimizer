from __future__ import annotations

import argparse
import logging

import uvicorn

from home_optimizer.app.container_factories import build_local_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.web import create_app

LOGGER = logging.getLogger(__name__)
LOCAL_DEFAULT_OVERRIDES = [
    "database_path=database.sqlite"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Home Optimizer locally.")
    parser.add_argument("--config", default="config.yaml", help="Path to local app config.")
    parser.add_argument(
        "--local-state",
        default="local.json",
        help="Path to local sensor state JSON.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config using dot notation, e.g. --set api_port=8100.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    LOGGER.info("Starting Home Optimizer web API locally")

    settings = load_settings(args.config, overrides=[*LOCAL_DEFAULT_OVERRIDES, *args.set])
    app = create_app(
        settings,
        container_factory=lambda app_settings: build_local_container(
            app_settings,
            local_state_path=args.local_state,
        ),
    )
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)


if __name__ == "__main__":
    main()
