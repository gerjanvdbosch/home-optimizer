from __future__ import annotations

from fastapi.testclient import TestClient

from home_optimizer.app.settings import AppSettings
from home_optimizer.features.history_import.schemas import HistoryImportResult
from home_optimizer.web.app import create_app


class FakeHomeAssistantGateway:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeHistoryImportService:
    def __init__(self, result: HistoryImportResult) -> None:
        self.result = result
        self.calls = 0

    def import_many(self, request) -> HistoryImportResult:
        self.calls += 1
        return self.result


class FakeContainer:
    def __init__(
        self,
        history_import_service: FakeHistoryImportService,
        home_assistant: FakeHomeAssistantGateway,
    ) -> None:
        self.history_import_service = history_import_service
        self.home_assistant = home_assistant


def test_dashboard_shows_import_button() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={"room_temperature": 3}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_enabled": True,
            "history_import_max_days_back": 14,
            "history_import_chunk_days": 2,
            "sensor_room_temperature": "sensor.room_temperature",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Importeer geschiedenis" in response.text
    assert "sensor.room_temperature" not in response.text
    assert gateway.closed is True


def test_settings_expose_typed_sensor_fields_without_options_bag() -> None:
    settings = AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "sensor_room_temperature": " sensor.room_temperature ",
            "sensor_outdoor_temperature": "",
        }
    )

    assert settings.sensor_room_temperature == "sensor.room_temperature"
    assert settings.sensor_outdoor_temperature is None
    assert not hasattr(settings, "options")


def test_history_import_endpoint_returns_summary() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(
        HistoryImportResult(imported_rows={"room_temperature": 3, "outdoor_temperature": 7})
    )
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_enabled": True,
            "history_import_max_days_back": 10,
            "history_import_chunk_days": 3,
            "sensor_room_temperature": "sensor.room_temperature",
            "sensor_outdoor_temperature": "sensor.outdoor_temperature",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "imported_rows": {
            "room_temperature": 3,
            "outdoor_temperature": 7,
        },
        "total_rows": 10,
        "sensor_count": 2,
    }
    assert service.calls == 1
    assert gateway.closed is True


def test_history_import_endpoint_rejects_disabled_import() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_enabled": False,
            "sensor_room_temperature": "sensor.room_temperature",
        }
    )
    app = create_app(
        settings,
        container_factory=lambda _: FakeContainer(
            history_import_service=service,
            home_assistant=gateway,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/api/history-import")

    assert response.status_code == 409
    assert response.json() == {"detail": "History import is uitgeschakeld."}
    assert service.calls == 0
