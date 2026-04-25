from __future__ import annotations

from home_optimizer.app.container import AppContainer, build_container
from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.infrastructure.local.gateway import LocalJsonGateway


def build_home_assistant_container(settings: AppSettings) -> AppContainer:
    return build_container(settings)


def build_local_container(
    settings: AppSettings,
    local_state_path: str = "local.json",
) -> AppContainer:
    def gateway_factory(specs: list[SensorSpec]) -> LocalJsonGateway:
        return LocalJsonGateway(local_state_path, specs)

    return build_container(
        settings,
        gateway_factory=gateway_factory,
        history_source="local_history",
        telemetry_source="local_telemetry",
    )
