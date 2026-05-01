from __future__ import annotations

from datetime import datetime
from typing import Protocol

from home_optimizer.domain import IdentifiedModel, NumericSeries, TextSeries
from home_optimizer.features.identification.schemas import IdentificationResult
from home_optimizer.features.history_import.schemas import HistoryImportRequest, HistoryImportResult
from home_optimizer.features.prediction.schemas import (
    RoomTemperaturePrediction,
    RoomTemperaturePredictionComparison,
)


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


class IdentificationRunner(Protocol):
    def identify(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult: ...

    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel: ...


class PredictionRunner(Protocol):
    def predict(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        thermostat_schedule: NumericSeries,
        shutter_schedule: NumericSeries | None = None,
    ) -> RoomTemperaturePrediction: ...

    def predict_vs_actual(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        thermostat_schedule: NumericSeries,
        shutter_schedule: NumericSeries | None = None,
    ) -> RoomTemperaturePredictionComparison: ...


class WebAppContainer(Protocol):
    @property
    def home_assistant(self) -> ClosableGateway: ...

    @property
    def history_import_service(self) -> HistoryImportRunner: ...

    @property
    def time_series_read_repository(self) -> DashboardDataReader: ...

    @property
    def identification_service(self) -> IdentificationRunner: ...

    @property
    def prediction_service(self) -> PredictionRunner: ...

    @property
    def telemetry_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def historical_weather_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def model_training_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def forecast_scheduler(self) -> TelemetrySchedulerRunner: ...

    @property
    def weather_import_service(self) -> WeatherImportRunner: ...

    def close(self) -> None: ...
