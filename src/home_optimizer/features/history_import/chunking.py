from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.features.history_import.models import ImportChunkWindow


class HistoryChunkPlanner:
    def __init__(self, chunk_days: int) -> None:
        if chunk_days <= 0:
            raise ValueError("chunk_days must be greater than zero")

        self.chunk_days = chunk_days

    def iter_windows(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[ImportChunkWindow]:
        windows: list[ImportChunkWindow] = []
        cursor = start_time

        while cursor < end_time:
            chunk_end = min(cursor + timedelta(days=self.chunk_days), end_time)
            windows.append(ImportChunkWindow(start_time=cursor, end_time=chunk_end))
            cursor = chunk_end

        return windows

