import logging
from datetime import UTC, datetime
from multiprocessing import Process, Queue

from joblib import parallel_backend

from app.forecasting import Forecasting
from domain.models.worker import Job, JobType


class Worker:
    def __init__(self, container_factory, manager):
        self.queue: Queue = Queue()
        self.container_factory = container_factory
        self.process: Process | None = None
        self.jobs = manager.dict()

    def start(self):
        if self.process and self.process.is_alive():
            return

        self.process = Process(
            target=self._run,
            daemon=True,
        )

        self.process.start()

    def submit(self, job: Job):
        self.jobs[job.id] = {
            "id": job.id,
            "type": job.type.value,
            "state": "queued",
            "created_at": datetime.now(UTC).isoformat(),
        }

        logging.info(
            "Job queued: id=%s type=%s",
            job.id,
            job.type.value,
        )

        self.queue.put(job)

    def stop(self):
        if not self.process:
            return

        self.queue.put(None)

        self.process.join(timeout=10)

        if self.process.is_alive():
            self.process.terminate()

    def _run(self):
        with parallel_backend("threading", n_jobs=1):
            container = self.container_factory()

            logging.info("Worker started")

            while True:
                job = self.queue.get()

                if job is None:
                    logging.info("Worker stopped")
                    break

                self._update_job(
                    job.id,
                    state="running",
                )

                logging.info(
                    "Job started: id=%s type=%s",
                    job.id,
                    job.type.value,
                )

                try:
                    self._execute(container, job)

                    logging.info(
                        "Job completed: id=%s type=%s",
                        job.id,
                        job.type.value,
                    )

                except Exception:
                    logging.exception(
                        "Job failed: id=%s type=%s",
                        job.id,
                        job.type.value,
                    )

                finally:
                    del self.jobs[job.id]

    def _update_job(self, job_id: str, **kwargs):
        job = dict(self.jobs[job_id])
        job.update(kwargs)
        self.jobs[job_id] = job

    def _execute(self, container, job: Job):
        config = container.config_repository.load()

        match job.type:
            case JobType.FIT:
                container.forecasting.fit(config, job.params)
            case JobType.PREDICT:
                container.forecasting.predict(config, job.params)
            case JobType.TUNE:
                container.forecasting.tune(config, job.params)
            case JobType.BACKTEST:
                container.forecasting.backtest(config, job.params)
            case _:
                raise NotImplementedError(f"Unknown job type={job.type}")
