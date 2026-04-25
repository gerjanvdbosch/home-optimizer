from __future__ import annotations

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import ForecastValue
from home_optimizer.infrastructure.database.session import Database


class ForecastRepository:
    def __init__(
        self,
        database: Database,
        source: str = "openmeteo",
    ) -> None:
        self.database = database
        self.source = source

    def write_entries(self, entries: list[ForecastEntry]) -> None:
        if not entries:
            return

        with self.database.session() as session:
            for entry in entries:
                session.merge(self._to_orm_value(entry))
            session.commit()

    def _to_orm_value(self, entry: ForecastEntry) -> ForecastValue:
        return ForecastValue(
            created_at_utc=normalize_utc_timestamp(entry.created_at_utc),
            forecast_time_utc=normalize_utc_timestamp(entry.forecast_time_utc),
            name=entry.name,
            value=entry.value,
            unit=entry.unit,
            source=entry.source,
        )
