from __future__ import annotations

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
