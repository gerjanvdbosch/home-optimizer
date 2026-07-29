import logging
from multiprocessing import Process, Queue

from domain.models.config import Job, JobType


class Worker:
    def __init__(self, container_factory):
        self.queue: Queue = Queue()
        self.container_factory = container_factory
        self.process: Process | None = None

    def start(self):
        if self.process and self.process.is_alive():
            return

        self.process = Process(
            target=self._run,
            daemon=True,
        )

        self.process.start()

    def submit(self, job: Job):
        self.queue.put(job)

    def stop(self):
        if not self.process:
            return

        self.queue.put(None)

        self.process.join(timeout=10)

        if self.process.is_alive():
            self.process.terminate()

    def _run(self):
        container = self.container_factory()

        logging.info("Worker started")

        while True:
            job = self.queue.get()

            if job is None:
                logging.info("Worker stopped")
                break

            try:
                self._execute(container, job)

            except Exception:
                logging.exception(
                    "Job failed: %s",
                    job.type,
                )

    def _execute(self, container, job: Job):
        config = container.config_repository.load()

        match job.type:
            case JobType.FIT:
                container.forecasting.fit(
                    config,
                    job.request,
                )

                logging.info("Fit finished")

            case JobType.TUNE:
                study = container.forecasting.tune(
                    config,
                    job.request,
                )

                logging.info(
                    "Tune finished %.3f %s",
                    study.best_value,
                    study.best_params,
                )

            case JobType.BACKTEST:
                metric = container.forecasting.backtest(
                    config,
                    job.request,
                )

                logging.info(
                    "Backtest finished %s",
                    metric,
                )
