import os
import threading
import logging
import uvicorn

from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler

from config import Config
from context import Context
from collector import Collector
from client import HAClient
from planner import Planner
from database import Database
from dhw import DhwMachine
from climate import ClimateMachine
from solar import SolarForecaster
from load import LoadForecaster
from webapi import api

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


class Coordinator:
    def __init__(self, context: Context, config: Config, database: Database, collector: Collector):
        self.solar = SolarForecaster(config, context, database)
        self.load = LoadForecaster(config, context, database)
        self.planner = Planner(context, config)
        self.dhw_machine = DhwMachine(context)
        self.climate_machine = ClimateMachine(context)
        self.context = context
        self.config = config
        self.collector = collector

    def tick(self):
        self.context.now = datetime.now(timezone.utc).replace(month=1, day=14, hour=10)

        self.collector.update_sensors()

        self.solar.update(self.context.now, self.context.stable_pv)
        self.load.update(self.context.now, self.context.stable_load)

        plan = self.planner.create_plan()

        self.dhw_machine.process(plan)
        self.climate_machine.process(plan)


    def train(self):
        self.solar.train()
        self.load.train()

    def start_api(self):
        api.state.coordinator = self
        uvicorn.run(
            api,
            host=self.config.webapi_host,
            port=self.config.webapi_port,
            log_level="warning",
        )


if __name__ == "__main__":
    logger.info("[System] Starting...")

    scheduler = BlockingScheduler()

    try:
        config = Config.load()
        context = Context(now=datetime.now(timezone.utc))
        client = HAClient(config)
        database = Database(config)
        collector = Collector(client, database, context, config)
        coordinator = Coordinator(context, config, database, collector)

        webapi = threading.Thread(target=coordinator.start_api, daemon=True)
        webapi.start()

        logger.info("[System] API server started")

        scheduler.add_job(collector.update_forecast, "interval", minutes=15)
        scheduler.add_job(collector.update_load, "interval", seconds=15)
        scheduler.add_job(collector.update_history, "interval", seconds=15)

        scheduler.add_job(coordinator.tick, "interval", seconds=5)
        scheduler.add_job(coordinator.train, "cron", hour=2, minute=5)

        logger.info("[System] Engine running")

        collector.update_forecast()
        collector.update_history()

        coordinator.tick()

        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("[System] Stopping and exiting...")
        scheduler.shutdown()
