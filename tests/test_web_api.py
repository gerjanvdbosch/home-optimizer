from __future__ import annotations

from time import sleep

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.sensor_factory import build_sensor_specs
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


class FakeLiveCollectionScheduler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeContainer:
    def __init__(
        self,
        history_import_service: FakeHistoryImportService,
        home_assistant: FakeHomeAssistantGateway,
    ) -> None:
        self.history_import_service = history_import_service
        self.home_assistant = home_assistant
        self.live_collection_scheduler = FakeLiveCollectionScheduler()


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(20):
        response = client.get(f"/api/history-import/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        sleep(0.05)

    raise AssertionError("history import job did not finish")


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
            "sensors": {"room_temperature": "sensor.room_temperature"},
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


def test_settings_reject_legacy_sensor_fields() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_options(
            {
                "database_path": "/tmp/home-optimizer-test.db",
                "sensor_room_temperature": "sensor.room_temperature",
            }
        )


def test_sensor_bindings_can_be_configured_as_mapping() -> None:
    settings = AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "sensors": {"room_temperature": " sensor.room_temperature "},
        }
    )

    specs = build_sensor_specs(settings)

    assert [spec.name for spec in specs] == ["room_temperature"]
    assert specs[0].entity_id == "sensor.room_temperature"


def test_sensor_bindings_reject_object_form() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_options(
            {
                "database_path": "/tmp/home-optimizer-test.db",
                "sensors": {"room_temperature": {"entity_id": "sensor.room_temperature"}},
            }
        )


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
            "sensors": {
                "room_temperature": "sensor.room_temperature",
                "outdoor_temperature": "sensor.outdoor_temperature",
            },
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
        job = wait_for_job(client, response.json()["job_id"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"pending", "running"}
    assert payload["sensor_count"] == 2
    assert payload["job_id"]
    assert job["status"] == "succeeded"
    assert service.calls == 1
    assert gateway.closed is True


def test_history_import_job_endpoint_returns_result() -> None:
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
            "sensors": {
                "room_temperature": "sensor.room_temperature",
                "outdoor_temperature": "sensor.outdoor_temperature",
            },
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
        payload = wait_for_job(client, response.json()["job_id"])

    assert payload["status"] == "succeeded"
    assert payload["imported_rows"] == {
        "room_temperature": 3,
        "outdoor_temperature": 7,
    }
    assert payload["total_rows"] == 10
    assert payload["sensor_count"] == 2
    assert payload["error"] is None


def test_history_import_endpoint_rejects_disabled_import() -> None:
    gateway = FakeHomeAssistantGateway()
    service = FakeHistoryImportService(HistoryImportResult(imported_rows={}))
    settings = AppSettings.from_options(
        {
            "api_port": 8099,
            "database_path": "/tmp/home-optimizer-test.db",
            "history_import_enabled": False,
            "sensors": {"room_temperature": "sensor.room_temperature"},
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
