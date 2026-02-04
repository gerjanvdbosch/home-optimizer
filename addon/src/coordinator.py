import os
import threading
import logging
import uvicorn

from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

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
        result = self.optimizer.resolve(self.context)

        if result is not None:
            self.context.result = result

            print(f"Status:  {result['status']}")
            print(
                f"Modus:   {result['mode']} (Vermogen: {result['target_power']:.2f} kW)"
            )
            print(f"Kosten:  €{result['cost_projected']:.2f}")
            print(f"Boiler:  {result['dhw_soc']*100:.1f}% SoC")

            print("\nVerloop komende 12 uur:")
            for i in range(48):
                r_t = result["planned_room"][i]
                d_t = result["planned_dhw"][i]
                print(f"  T + {i*15:02}m | Kamer: {r_t:.2f}°C | Boiler: {d_t:.2f}°C")

    def train(self):
        self.solar.train()
        self.load.train()
        self.optimizer.train()

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

        next_run = datetime.now(timezone.utc) + timedelta(seconds=10)

        scheduler.add_job(
            collector.update_forecast, "interval", minutes=15, next_run_time=next_run
        )
        scheduler.add_job(
            collector.update_load, "interval", seconds=5, next_run_time=next_run
        )
        scheduler.add_job(
            collector.update_history, "interval", minutes=1, next_run_time=next_run
        )

        scheduler.add_job(
            coordinator.tick, "interval", minutes=1, next_run_time=next_run
        )
        scheduler.add_job(
            coordinator.optimize, "interval", minutes=15, next_run_time=next_run
        )
        scheduler.add_job(coordinator.train, "cron", hour=2, minute=5)

        logger.info("[System] Engine running")

        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("[System] Stopping and exiting...")
        scheduler.shutdown()
