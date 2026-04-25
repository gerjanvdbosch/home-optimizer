from __future__ import annotations

import logging
from datetime import datetime

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.history_import.chunking import HistoryChunkPlanner
from home_optimizer.features.history_import.history_mapping import map_history_points
from home_optimizer.features.history_import.ports import HistoryRepository, HistorySourceGateway
from home_optimizer.features.history_import.resampling import MinuteResampler
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult

LOGGER = logging.getLogger(__name__)


class HistoryImportService:
    def __init__(
        self,
        gateway: HistorySourceGateway,
        repository: HistoryRepository,
        chunk_days: int = 3,
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.chunk_planner = HistoryChunkPlanner(chunk_days)
        self.resampler = MinuteResampler(repository.source)

    def import_many(self, request: HistoryImportRequest) -> HistoryImportResult:
        results: dict[str, int] = {}

        for spec in request.specs:
            results[spec.name] = self.import_sensor(
                spec=spec,
                start_time=request.start_time,
                end_time=request.end_time,
            )

        return HistoryImportResult(imported_rows=results)

    def import_sensor(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> int:
        start = ensure_utc(start_time)
        end = ensure_utc(end_time or datetime.now(start.tzinfo))

        total_written = 0
        carry_value = self.repository.last_stored_value_before(spec, start)

        for window in self.chunk_planner.iter_windows(start, end):
            if self.repository.chunk_already_imported(spec, window.start_time, window.end_time):
                LOGGER.info(
                    "Skip %s: %s to %s already imported",
                    spec.name,
                    window.start_time.isoformat(),
                    window.end_time.isoformat(),
                )
                carry_value = self.repository.last_stored_value_before(spec, window.end_time)
                continue

            history = self.gateway.get_history(
                entity_id=spec.entity_id,
                start_time=window.start_time,
                end_time=window.end_time,
                minimal_response=True,
            )
            points = map_history_points(history, spec)
            samples = self.resampler.resample(
                points=points,
                spec=spec,
                start_time=window.start_time,
                end_time=window.end_time,
                initial_value=carry_value,
            )
            carry_value = points[-1].value if points else carry_value

            written = self.repository.write_new_samples(samples)
            if self._is_stable_window(window.end_time):
                self.repository.mark_chunk_imported(
                    spec,
                    window.start_time,
                    window.end_time,
                    written,
                )
            total_written += written

        return total_written

    @staticmethod
    def _is_stable_window(end_time: datetime, now: datetime | None = None) -> bool:
        current_time = ensure_utc(now or datetime.now(end_time.tzinfo))
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        return end_time <= today_start
