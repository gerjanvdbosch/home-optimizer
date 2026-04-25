from __future__ import annotations

from dataclasses import dataclass

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.features.history_import.service import HistoryImportService
from home_optimizer.features.live_collection.service import LiveCollectionService
from home_optimizer.infrastructure.database.session import Database
from home_optimizer.infrastructure.database.timeseries_repository import TimeSeriesRepository
from home_optimizer.infrastructure.home_assistant.gateway import HomeAssistantGateway


@dataclass
class AppContainer:
    settings: AppSettings
    database: Database
    home_assistant: HomeAssistantGateway
    history_import_repository: TimeSeriesRepository
    history_import_service: HistoryImportService
    live_collection_repository: TimeSeriesRepository
    live_collection_service: LiveCollectionService


def build_container(settings: AppSettings) -> AppContainer:
    database = Database(settings.database_path)
    database.init_schema()

    home_assistant = HomeAssistantGateway()
    history_import_repository = TimeSeriesRepository(database, source="home_assistant_history")
    live_collection_repository = TimeSeriesRepository(database, source="home_assistant_live")
    sensor_specs = build_sensor_specs(settings)
    history_import_service = HistoryImportService(
        gateway=home_assistant,
        repository=history_import_repository,
        chunk_days=settings.history_import_chunk_days,
    )
    live_collection_service = LiveCollectionService(
        gateway=home_assistant,
        repository=live_collection_repository,
        specs=sensor_specs,
    )

    return AppContainer(
        settings=settings,
        database=database,
        home_assistant=home_assistant,
        history_import_repository=history_import_repository,
        history_import_service=history_import_service,
        live_collection_repository=live_collection_repository,
        live_collection_service=live_collection_service,
    )
