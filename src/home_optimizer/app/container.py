from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from home_optimizer.app.ports import SensorGateway
from home_optimizer.app.settings import AppSettings
from home_optimizer.app.telemetry_scheduler import TelemetryScheduler
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.features.history_import.service import HistoryImportService
from home_optimizer.features.telemetry.service import TelemetryService
from home_optimizer.infrastructure.database.session import Database
from home_optimizer.infrastructure.database.timeseries_repository import TimeSeriesRepository
from home_optimizer.infrastructure.home_assistant.gateway import HomeAssistantGateway

GatewayFactory = Callable[[list[SensorSpec]], SensorGateway]


@dataclass
class AppContainer:
    settings: AppSettings
    database: Database
    home_assistant: SensorGateway
    history_import_repository: TimeSeriesRepository
    history_import_service: HistoryImportService
    telemetry_repository: TimeSeriesRepository
    telemetry_service: TelemetryService
    telemetry_scheduler: TelemetryScheduler


def build_container(
    settings: AppSettings,
    gateway_factory: GatewayFactory | None = None,
    history_source: str = "home_assistant_history",
    telemetry_source: str = "home_assistant_telemetry",
) -> AppContainer:
    database = Database(settings.database_path)
    database.init_schema()

    sensor_specs = build_sensor_specs(settings)
    gateway = gateway_factory(sensor_specs) if gateway_factory else HomeAssistantGateway()
    history_import_repository = TimeSeriesRepository(database, source=history_source)
    telemetry_repository = TimeSeriesRepository(database, source=telemetry_source)
    history_import_service = HistoryImportService(
        gateway=gateway,
        repository=history_import_repository,
        chunk_days=settings.history_import_chunk_days,
    )
    telemetry_service = TelemetryService(
        gateway=gateway,
        repository=telemetry_repository,
        specs=sensor_specs,
    )
    telemetry_scheduler = TelemetryScheduler(telemetry_service)

    return AppContainer(
        settings=settings,
        database=database,
        home_assistant=gateway,
        history_import_repository=history_import_repository,
        history_import_service=history_import_service,
        telemetry_repository=telemetry_repository,
        telemetry_service=telemetry_service,
        telemetry_scheduler=telemetry_scheduler,
    )
