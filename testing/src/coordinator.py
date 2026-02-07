import os
import logging
import uvicorn

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
        self.database = database

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

        next_run = datetime.now(timezone.utc) + timedelta(seconds=5)

        scheduler.add_job(
            collector.update_forecast, "interval", seconds=15, id="update_forecast"
        )
        scheduler.add_job(
            collector.update_load, "interval", seconds=15, id="update_load"
        )
        scheduler.add_job(
            collector.update_history, "interval", seconds=15, id="update_history"
        )
        scheduler.add_job(coordinator.tick, "interval", seconds=15, id="tick")

        scheduler.add_job(coordinator.train, "cron", hour=2, minute=5, id="train")
        scheduler.add_job(
            coordinator.optimize,
            "interval",
            minutes=1,
            next_run_time=next_run,
            id="optimize",
        )

        collector.update_forecast()
        collector.update_history()

        coordinator.tick()
        coordinator.train()

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
