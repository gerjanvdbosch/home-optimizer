from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from home_optimizer.app.history_import_requests import build_history_import_request
from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.models import DomainModel
from home_optimizer.features.history.schemas import HistoryImportResult


class HistoryImportRunner(Protocol):
    def import_many(self, request) -> HistoryImportResult: ...


class HistoryImportJob(DomainModel):
    job_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    imported_rows: dict[str, int] = Field(default_factory=dict)
    total_rows: int = 0
    sensor_count: int = 0
    error: str | None = None


class HistoryImportJobRunner:
    def __init__(
        self,
        settings: AppSettings,
        importer: HistoryImportRunner,
        max_workers: int = 1,
    ) -> None:
        self.settings = settings
        self.importer = importer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, HistoryImportJob] = {}
        self.futures: dict[str, Future[None]] = {}
        self.lock = Lock()

    def start(self) -> HistoryImportJob:
        request = build_history_import_request(self.settings)
        job = HistoryImportJob(
            job_id=uuid4().hex,
            status="pending",
            created_at=utc_now(),
            sensor_count=len(request.specs),
        )

        with self.lock:
            self.jobs[job.job_id] = job
            self.futures[job.job_id] = self.executor.submit(self._run, job.job_id, request)

        return job

    def get(self, job_id: str) -> HistoryImportJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, request) -> None:
        self._update(job_id, status="running", started_at=utc_now())
        try:
            result = self.importer.import_many(request)
            total_rows = sum(result.imported_rows.values())
            self._update(
                job_id,
                status="succeeded",
                finished_at=utc_now(),
                imported_rows=result.imported_rows,
                total_rows=total_rows,
                sensor_count=len(request.specs),
            )
        except Exception as exc:  # pragma: no cover - exercised through API integration failures.
            self._update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=str(exc),
                sensor_count=len(request.specs),
            )

    def _update(self, job_id: str, **changes) -> None:
        with self.lock:
            job = self.jobs[job_id]
            self.jobs[job_id] = job.model_copy(update=changes)
