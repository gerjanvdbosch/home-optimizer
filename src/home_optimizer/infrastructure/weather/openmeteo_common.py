from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

from home_optimizer.domain.names import (
    FORECAST_DEW_POINT,
    FORECAST_DIFFUSE_RADIATION,
    FORECAST_DIRECT_RADIATION,
    FORECAST_HUMIDITY,
    FORECAST_TEMPERATURE,
    FORECAST_WIND,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_PV,
)
from home_optimizer.domain.time import ensure_utc

GTI_VARIABLE = "global_tilted_irradiance"
LIVING_ROOM_WINDOW_TILT = 90.0

OPEN_METEO_BASE_VARIABLES = {
    FORECAST_TEMPERATURE: "temperature_2m",
    FORECAST_HUMIDITY: "relative_humidity_2m",
    FORECAST_WIND: "wind_speed_10m",
    FORECAST_DEW_POINT: "dew_point_2m",
    FORECAST_DIRECT_RADIATION: "direct_radiation",
    FORECAST_DIFFUSE_RADIATION: "diffuse_radiation",
}
OPEN_METEO_WEATHER_UNITS = {
    FORECAST_TEMPERATURE: "degC",
    FORECAST_HUMIDITY: "%",
    FORECAST_WIND: "ms",
    FORECAST_DEW_POINT: "degC",
    FORECAST_DIRECT_RADIATION: "Wm2",
    FORECAST_DIFFUSE_RADIATION: "Wm2",
    GTI_PV: "Wm2",
    GTI_LIVING_ROOM_WINDOWS: "Wm2",
}

TEntry = TypeVar("TEntry")


def build_entries_from_payload(
    *,
    payload: dict[str, object],
    section_name: str,
    variable_map: dict[str, str],
    parse_timestamp: Callable[[object], datetime],
    entry_factory: Callable[[datetime, str, float], TEntry],
    missing_payload_message: str,
    missing_timestamp_message: str,
    missing_variable_message: str,
    length_mismatch_message: str,
) -> list[TEntry]:
    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(missing_payload_message)

    raw_times = section.get("time")
    if not isinstance(raw_times, list):
        raise ValueError(missing_timestamp_message)

    times = [parse_timestamp(value) for value in raw_times]
    entries: list[TEntry] = []

    for name, variable in variable_map.items():
        raw_values = section.get(variable)
        if not isinstance(raw_values, list):
            raise ValueError(missing_variable_message.format(variable=variable))
        if len(raw_values) != len(times):
            raise ValueError(length_mismatch_message.format(variable=variable))

        for timestamp, value in zip(times, raw_values, strict=True):
            if value is None:
                continue
            entries.append(entry_factory(timestamp, name, float(value)))

    return entries


def parse_hourly_open_meteo_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo timestamp must be a string")
    return ensure_utc(datetime.fromisoformat(f"{value}+00:00"))


def parse_minutely_open_meteo_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo timestamp must be a string")
    return ensure_utc(datetime.fromisoformat(f"{value}:00+00:00"))


def compass_to_open_meteo_azimuth(compass_degrees: float) -> float:
    converted = compass_degrees - 180.0
    if converted > 180.0:
        converted -= 360.0
    if converted <= -180.0:
        converted += 360.0
    return converted


def build_gti_entries(
    *,
    pv_tilt: float | None,
    pv_azimuth: float | None,
    living_room_window_azimuth: float | None,
    fetch_payload: Callable[[float, float], dict[str, object]],
    entries_from_payload: Callable[[dict[str, object], dict[str, str]], list[TEntry]],
) -> list[TEntry]:
    entries: list[TEntry] = []

    if pv_tilt is not None and pv_azimuth is not None:
        entries.extend(
            entries_from_payload(
                fetch_payload(pv_tilt, compass_to_open_meteo_azimuth(pv_azimuth)),
                {GTI_PV: GTI_VARIABLE},
            )
        )

    if living_room_window_azimuth is not None:
        entries.extend(
            entries_from_payload(
                fetch_payload(
                    LIVING_ROOM_WINDOW_TILT,
                    compass_to_open_meteo_azimuth(living_room_window_azimuth),
                ),
                {GTI_LIVING_ROOM_WINDOWS: GTI_VARIABLE},
            )
        )

    return entries
