from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import NumericSeries, TextSeries
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult


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


class TelemetrySchedulerRunner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def dashboard_repository(self) -> DashboardDataReader: ...

    @property
    def telemetry_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def forecast_scheduler(self) -> TelemetrySchedulerRunner: ...

    def close(self) -> None: ...
