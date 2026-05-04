from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import NumericSeries, TextSeries
from home_optimizer.features.history.schemas import HistoryImportRequest, HistoryImportResult


class ClosableGateway(Protocol):
    def close(self) -> None: ...


class HistoryImportRunner(Protocol):
    def import_many(self, request: HistoryImportRequest) -> HistoryImportResult: ...


class DashboardDataReader(Protocol):
    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[TextSeries]: ...

    def read_forecast_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...

    def read_historical_weather_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]: ...


class TelemetrySchedulerRunner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class WeatherImportRunner(Protocol):
    def import_weather_data(
        self,
        created_at: datetime | None = None,
    ) -> int: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def time_series_read_repository(self) -> DashboardDataReader: ...

    @property
    def telemetry_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def historical_weather_scheduler(self) -> TelemetrySchedulerRunner: ...


    @property
    def electricity_price_scheduler(self) -> TelemetrySchedulerRunner: ...


    @property
    def forecast_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def weather_import_service(self) -> WeatherImportRunner: ...

    def close(self) -> None: ...
