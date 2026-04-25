from __future__ import annotations

from datetime import datetime, timezone


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Datetime must include a timezone: {value}")
    return dt


def ensure_utc(value: datetime | str) -> datetime:
    dt = parse_datetime(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)


def normalize_utc_timestamp(
    value: datetime | str,
    *,
    timespec: str = "seconds",
) -> str:
    return ensure_utc(value).isoformat(timespec=timespec)
