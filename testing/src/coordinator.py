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
        self.context.now = datetime.now(timezone.utc).replace(
            day=14, month=1, year=2026, hour=12
        )

        self.collector.update_sensors()

        df = self.context.forecast_df_raw.copy()
        df = self.solar.update(df)
        df = self.load.update(df)

        self.context.forecast_df = df

    def optimize(self):
        result = self.optimizer.resolve(self.context)

        if result:
            logger.info("--- Resultaat ---")
            logger.info(f"Gekozen Modus: {result['mode']}")
            logger.info(f"Frequentie: {result['freq']} Hz")

            # Toegang tot de interne solver variabelen via de mpc instantie
            mpc = self.optimizer.mpc

            if mpc.f_ufh.value is not None:
                logger.info("\nPlan details:")
                f_u = mpc.f_ufh.value
                f_d = mpc.f_dhw.value
                t_r = mpc.t_room.value
                t_d = mpc.t_dhw.value  # De voorspelde DHW temp

                tz = datetime.now().astimezone().tzinfo

                plan_data = []
                # We lopen door de hele horizon (48 kwartieren / 12 uur)
                for i in range(mpc.horizon):
                    mode = "UFH" if f_u[i] > 5 else "DHW" if f_d[i] > 5 else "OFF"
                    plan_data.append(
                        {
                            "Tijd": self.context.forecast_df["timestamp"]
                            .iloc[i]
                            .tz_convert(tz)
                            .strftime("%H:%M"),
                            "Mode": mode,
                            "Freq": round(max(f_u[i], f_d[i]), 1),
                            "T_room": round(t_r[i], 2),
                            "T_dhw": round(t_d[i], 2),
                        }
                    )

                # Gebruik pandas om een mooie tabel te printen
                print(pd.DataFrame(plan_data).to_string(index=False))

            # Sla het resultaat op in de context voor de Web API
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
        context = Context(
            now=datetime.now(timezone.utc).replace(day=14, month=1, year=2026, hour=12)
        )
        client = HAClient(config)
        database = Database(config)
        collector = Collector(client, database, context, config)
        coordinator = Coordinator(context, config, database, collector)

        api.state.coordinator = coordinator

        next_run = datetime.now(timezone.utc) + timedelta(seconds=3)

        scheduler.add_job(
            collector.update_forecast, "interval", seconds=15, id="forecast"
        )
        scheduler.add_job(collector.update_load, "interval", seconds=15, id="load")
        scheduler.add_job(
            collector.update_history, "interval", seconds=15, id="history"
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
