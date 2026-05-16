from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta

from .forecast import ForecastEntry
from .time import ensure_utc


def latest_forecast_entries(entries: list[ForecastEntry]) -> list[ForecastEntry]:
    latest_by_name_and_time: dict[tuple[str, datetime], ForecastEntry] = {}
    for entry in entries:
        forecast_time_utc = ensure_utc(entry.forecast_time_utc)
        key = (entry.name, forecast_time_utc)
        existing = latest_by_name_and_time.get(key)
        if existing is None or ensure_utc(entry.created_at_utc) >= ensure_utc(existing.created_at_utc):
            latest_by_name_and_time[key] = entry
    return sorted(
        latest_by_name_and_time.values(),
        key=lambda entry: (entry.name, ensure_utc(entry.forecast_time_utc)),
    )


def resample_forecast_entries(
    entries: list[ForecastEntry],
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
    max_interpolation_gap_minutes: int = 30,
) -> list[ForecastEntry]:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")

    start_time_utc = ensure_utc(start_time)
    end_time_utc = ensure_utc(end_time)
    if end_time_utc <= start_time_utc:
        return []

    latest_entries = latest_forecast_entries(entries)
    if not latest_entries:
        return []

    interval = timedelta(minutes=interval_minutes)
    resampled_entries: list[ForecastEntry] = []
    entries_by_name: dict[str, list[ForecastEntry]] = {}
    for entry in latest_entries:
        entries_by_name.setdefault(entry.name, []).append(entry)

    for name, name_entries in entries_by_name.items():
        timestamps = [ensure_utc(entry.forecast_time_utc) for entry in name_entries]
        cursor = start_time_utc
        while cursor < end_time_utc:
            exact_index = bisect_left(timestamps, cursor)
            exact_entry = (
                name_entries[exact_index]
                if exact_index < len(name_entries) and timestamps[exact_index] == cursor
                else None
            )
            if exact_entry is not None:
                resampled_entries.append(
                    ForecastEntry(
                        created_at_utc=ensure_utc(exact_entry.created_at_utc),
                        forecast_time_utc=cursor,
                        name=name,
                        value=float(exact_entry.value),
                        unit=exact_entry.unit,
                        source=exact_entry.source,
                    )
                )
                cursor += interval
                continue

            if exact_index == 0 or exact_index >= len(name_entries):
                cursor += interval
                continue

            left_entry = name_entries[exact_index - 1]
            right_entry = name_entries[exact_index]
            left_time = ensure_utc(left_entry.forecast_time_utc)
            right_time = ensure_utc(right_entry.forecast_time_utc)
            gap_minutes = (right_time - left_time).total_seconds() / 60.0
            if gap_minutes > max_interpolation_gap_minutes:
                cursor += interval
                continue
            span_seconds = (right_time - left_time).total_seconds()
            if span_seconds <= 0:
                cursor += interval
                continue

            if left_entry.unit is None or right_entry.unit is None:
                resampled_value = float(left_entry.value)
            else:
                progress = (cursor - left_time).total_seconds() / span_seconds
                resampled_value = float(left_entry.value) + (
                    (float(right_entry.value) - float(left_entry.value)) * progress
                )
            resampled_entries.append(
                ForecastEntry(
                    created_at_utc=max(
                        ensure_utc(left_entry.created_at_utc),
                        ensure_utc(right_entry.created_at_utc),
                    ),
                    forecast_time_utc=cursor,
                    name=name,
                    value=resampled_value,
                    unit=left_entry.unit or right_entry.unit,
                    source=right_entry.source,
                )
            )
            cursor += interval

    return sorted(
        resampled_entries,
        key=lambda entry: (entry.name, ensure_utc(entry.forecast_time_utc)),
    )
