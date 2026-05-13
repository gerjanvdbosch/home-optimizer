from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from home_optimizer.domain import NumericSeries, TextSeries
from home_optimizer.features.history.schemas import HistoryImportRequest, HistoryImportResult
from home_optimizer.features.modeling import StoredModelVersion
from home_optimizer.features.mpc import MpcPlan


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

    def read_electricity_price_series(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        source: str,
        interval_minutes: int = 15,
    ) -> NumericSeries: ...


class TelemetrySchedulerRunner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class WeatherImportRunner(Protocol):
    def import_weather_data(
        self,
        created_at: datetime | None = None,
    ) -> int: ...


class ModelVersionWriter(Protocol):
    def save_room_model_version(self, version: StoredModelVersion) -> None: ...

    def get_room_model_version(self, model_id: str) -> StoredModelVersion | None: ...

    def get_active_room_model_version(self) -> StoredModelVersion | None: ...

    def list_room_model_versions(self) -> list[object]: ...


class SpaceHeatingMpcPlanningRunner(Protocol):
    def plan(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int | None = None,
        horizon_steps: int = 36,
        default_effective_heating_kw: float | None = None,
        max_solver_seconds: float | None = None,
    ) -> MpcPlan: ...


class DatasetFrameReader(Protocol):
    def read_samples(
        self,
        *,
        interval_minutes: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def time_series_read_repository(self) -> DashboardDataReader: ...

    @property
    def dataset_repository(self) -> DatasetFrameReader: ...

    @property
    def telemetry_scheduler(self) -> TelemetrySchedulerRunner: ...



    @property
    def electricity_price_scheduler(self) -> TelemetrySchedulerRunner: ...


    @property
    def forecast_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def weather_import_service(self) -> WeatherImportRunner: ...

    @property
    def model_version_repository(self) -> ModelVersionWriter: ...

    @property
    def space_heating_mpc_planning_service(self) -> SpaceHeatingMpcPlanningRunner: ...

    def close(self) -> None: ...
