from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import ensure_utc
from home_optimizer.domain.timeseries import MinuteSample


class LiveMinuteBuffer:
    def __init__(self, source: str) -> None:
        self.source = source
        self._values: dict[tuple[str, datetime], list[Any]] = defaultdict(list)
        self._specs: dict[str, SensorSpec] = {}

    def add(self, spec: SensorSpec, timestamp: datetime, value: Any) -> None:
        minute = self._floor_minute(ensure_utc(timestamp))
        self._specs[spec.name] = spec
        self._values[(spec.name, minute)].append(value)

    def pop_samples_before(self, before: datetime | None = None) -> list[MinuteSample]:
        before_utc = ensure_utc(before) if before else None
        samples: list[MinuteSample] = []

        for key in sorted(self._values.keys(), key=lambda item: (item[1], item[0])):
            name, minute = key
            if before_utc is not None and minute >= before_utc:
                continue

            values = self._values.pop(key)
            sample = self._build_sample(self._specs[name], minute, values)
            if sample:
                samples.append(sample)

        return samples

    def has_samples(self) -> bool:
        return bool(self._values)

    def _build_sample(
        self,
        spec: SensorSpec,
        minute: datetime,
        values: list[Any],
    ) -> MinuteSample | None:
        if not values:
            return None

        numeric_values = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        last_value = values[-1]

        mean_real = None
        min_real = None
        max_real = None
        last_real = None
        last_text = None
        last_bool = None

        if numeric_values:
            mean_real = sum(numeric_values) / len(numeric_values)
            min_real = min(numeric_values)
            max_real = max(numeric_values)

        if isinstance(last_value, bool):
            last_bool = int(last_value)
        elif isinstance(last_value, (int, float)):
            last_real = float(last_value)
        else:
            last_text = str(last_value)

        return MinuteSample(
            timestamp_minute=minute,
            name=spec.name,
            source=self.source,
            entity_id=spec.entity_id,
            category=spec.category,
            unit=spec.unit,
            mean_real=mean_real,
            min_real=min_real,
            max_real=max_real,
            last_real=last_real,
            last_text=last_text,
            last_bool=last_bool,
            sample_count=len(values),
        )

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)

