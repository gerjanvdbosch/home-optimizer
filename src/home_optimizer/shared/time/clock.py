from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def utc_now(clock: Clock | None = None) -> datetime:
    active_clock = clock or SystemClock()
    return active_clock.now()
