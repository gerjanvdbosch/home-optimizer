from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from home_optimizer.app.live_collection_scheduler import LiveCollectionScheduler
from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.features.history_import.service import HistoryImportService
from home_optimizer.features.live_collection.service import LiveCollectionService
from home_optimizer.infrastructure.database.session import Database
from home_optimizer.infrastructure.database.timeseries_repository import TimeSeriesRepository
from home_optimizer.infrastructure.home_assistant.gateway import HomeAssistantGateway


class AppGateway(Protocol):
    def close(self) -> None: ...

    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def get_states(self) -> list[dict[str, Any]]: ...

    def get_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        minimal_response: bool = True,
    ) -> list[dict[str, Any]]: ...


GatewayFactory = Callable[[list[SensorSpec]], AppGateway]


@dataclass
class AppContainer:
    settings: AppSettings
    database: Database
    home_assistant: AppGateway
    history_import_repository: TimeSeriesRepository
    history_import_service: HistoryImportService
    live_collection_repository: TimeSeriesRepository
    live_collection_service: LiveCollectionService
    live_collection_scheduler: LiveCollectionScheduler


def build_container(
    settings: AppSettings,
    gateway_factory: GatewayFactory | None = None,
    history_source: str = "home_assistant_history",
    live_source: str = "home_assistant_live",
) -> AppContainer:
    database = Database(settings.database_path)
    database.init_schema()

    sensor_specs = build_sensor_specs(settings)
    gateway = gateway_factory(sensor_specs) if gateway_factory else HomeAssistantGateway()
    history_import_repository = TimeSeriesRepository(database, source=history_source)
    live_collection_repository = TimeSeriesRepository(database, source=live_source)
    history_import_service = HistoryImportService(
        gateway=gateway,
        repository=history_import_repository,
        chunk_days=settings.history_import_chunk_days,
    )
    live_collection_service = LiveCollectionService(
        gateway=gateway,
        repository=live_collection_repository,
        specs=sensor_specs,
    )
    live_collection_scheduler = LiveCollectionScheduler(live_collection_service)

    return AppContainer(
        settings=settings,
        database=database,
        home_assistant=gateway,
        history_import_repository=history_import_repository,
        history_import_service=history_import_service,
        live_collection_repository=live_collection_repository,
        live_collection_service=live_collection_service,
        live_collection_scheduler=live_collection_scheduler,
    )
