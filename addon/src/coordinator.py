import os
import logging
import uvicorn
import pandas as pd

from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from config import Config
from context import Context
from collector import Collector
from client import HAClient
from database import Database
from solar import SolarForecaster
from load import LoadForecaster
from optimizer import Optimizer
from webapi import api

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


class Coordinator:
    def __init__(
        self, context: Context, config: Config, database: Database, collector: Collector
    ):
        self.solar = SolarForecaster(config, context, database)
        self.load = LoadForecaster(config, context, database)
        self.optimizer = Optimizer(config, database)
        self.context = context
        self.config = config
        self.collector = collector

    def tick(self):
        self.context.now = datetime.now(timezone.utc)

        self.collector.update_sensors()

        df = self.context.forecast_df_raw.copy()
        df = self.solar.update(df)
        df = self.load.update(df)

        self.context.forecast_df = df

    def optimize(self):
        # Stap 1: Los de MPC op via de optimizer
        result = self.optimizer.resolve(self.context)

        if result:
            logger.info("--- Resultaat ---")
            logger.info(f"Status: {result.get('status', 'FAIL')}")
            logger.info(f"Gekozen Modus: {result.get('mode', 'OFF')}")

            if result.get("mode") != "OFF":
                logger.info(
                    f"Doel Elektrisch Vermogen: {result.get('target_pel_kw')} kW"
                )
                logger.info(f"Doel Aanvoertemp: {result.get('target_supply_temp')} °C")

            # Print tabel in de console
            print(pd.DataFrame(result.get("plan")).to_string(index=False))

        # Sla het plan op in de context voor de UI of verdere verwerking
        self.context.result = result

    def train(self):
        self.solar.train()
        self.load.train()
        self.optimizer.train()


if __name__ == "__main__":
    logger.info("[System] Starting...")

    scheduler = BackgroundScheduler(job_defaults={"max_instances": 1})

    try:
        config = Config.load()
        context = Context(now=datetime.now(timezone.utc))
        client = HAClient(config)
        database = Database(config)
        collector = Collector(client, database, context, config)
        coordinator = Coordinator(context, config, database, collector)

        api.state.coordinator = coordinator

        next_run = datetime.now(timezone.utc) + timedelta(seconds=10)

        scheduler.add_job(
            collector.update_forecast, "interval", minutes=15, id="forecast"
        )
        scheduler.add_job(collector.update_load, "interval", seconds=5, id="load")
        scheduler.add_job(collector.update_history, "interval", minutes=1, id="history")

        scheduler.add_job(coordinator.tick, "interval", minutes=1, id="tick")
        scheduler.add_job(
            coordinator.optimize,
            "interval",
            minutes=15,
            next_run_time=next_run,
            id="optimize",
        )
        scheduler.add_job(coordinator.train, "cron", hour=2, minute=5, id="train")

        logger.info("[System] Engine running")

        collector.update_forecast()
        collector.update_history()

        coordinator.tick()

        scheduler.start()

        logger.info("[System] Engine running")

        uvicorn.run(
            api,
            host=config.webapi_host,
            port=config.webapi_port,
            log_level="warning",
        )

    except (KeyboardInterrupt, SystemExit):
        logger.info("[System] Stopping and exiting...")
        scheduler.shutdown()
