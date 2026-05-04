from __future__ import annotations

from datetime import time

import pytest
from pydantic import ValidationError

from home_optimizer.app import AppSettings
from home_optimizer.app.settings_loader import deep_merge, parse_dot_overrides


def test_parse_dot_overrides_builds_nested_options() -> None:
    assert parse_dot_overrides(
        [
            "api_port=8100",
            "history_import_max_days_back=14",
            "sensors.room_temperature=sensor.local_room",
        ]
    ) == {
        "api_port": 8100,
        "history_import_max_days_back": 14,
        "sensors": {"room_temperature": "sensor.local_room"},
    }


def test_deep_merge_preserves_unrelated_nested_values() -> None:
    merged = deep_merge(
        {
            "api_port": 8099,
            "sensors": {
                "room_temperature": "sensor.room",
                "outdoor_temperature": "sensor.outdoor",
            },
        },
        {
            "sensors": {
                "room_temperature": "sensor.local_room",
            }
        },
    )

    assert merged == {
        "api_port": 8099,
        "sensors": {
            "room_temperature": "sensor.local_room",
            "outdoor_temperature": "sensor.outdoor",
        },
    }


def test_app_settings_temperature_target_schedule() -> None:
    settings = AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "room_target": [
                {"time": "18:00", "target": 20.0, "low_margin": 0.5, "high_margin": 1.5},
                {"time": "08:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            ],
        }
    )

    assert [(window.time, window.target) for window in settings.room_target] == [
        (time(8, 0), 19.0),
        (time(18, 0), 20.0),
    ]

    with pytest.raises(ValidationError):
        AppSettings.from_options(
            {
                "database_path": "/tmp/home-optimizer-test.db",
                "room_target": [
                    {"time": "20:00", "target": 20.0, "low_margin": 0.5, "high_margin": 1.5},
                    {"time": "20:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
                ],
            }
        )

