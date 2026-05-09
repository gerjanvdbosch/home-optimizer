from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from home_optimizer.features.dataset.models import MpcDatasetRow


@dataclass(frozen=True)
class PreparedRoomData:
    timestamps_utc: list[datetime]
    field_arrays: dict[str, np.ndarray]
    segment_masks: dict[str, np.ndarray]


def _sanitize_nonfinite(values: np.ndarray, *, fallback: float) -> np.ndarray:
    sanitized = values.astype(float, copy=True)
    sanitized[~np.isfinite(sanitized)] = fallback
    return sanitized


def room_identification_mask(rows: list[MpcDatasetRow]) -> np.ndarray:
    return np.asarray([row.is_valid_for_room_identification for row in rows], dtype=bool)


def prepare_room_data(rows: list[MpcDatasetRow], config) -> PreparedRoomData:
    timestamps_utc = [row.timestamp_utc for row in rows]
    local_hours = np.asarray(
        [timestamp_utc.astimezone().hour for timestamp_utc in timestamps_utc],
        dtype=int,
    )
    outdoor_temperature = np.asarray(
        [
            np.nan if row.outdoor_temperature_c is None else float(row.outdoor_temperature_c)
            for row in rows
        ],
        dtype=float,
    )
    thermal_output = np.asarray(
        [
            0.0
            if row.thermal_output_estimate_kw is None
            else float(row.thermal_output_estimate_kw)
            for row in rows
        ],
        dtype=float,
    )
    solar_gain = np.asarray(
        [
            0.0 if row.solar_gain_proxy_w_m2 is None else float(row.solar_gain_proxy_w_m2)
            for row in rows
        ],
        dtype=float,
    )
    solar_irradiance = np.asarray(
        [
            0.0 if row.solar_irradiance_w_m2 is None else float(row.solar_irradiance_w_m2)
            for row in rows
        ],
        dtype=float,
    )
    shutter_position = np.asarray(
        [0.0 if row.shutter_position_pct is None else float(row.shutter_position_pct) for row in rows],
        dtype=float,
    )
    occupied_flag = np.asarray(
        [0.0 if row.occupied_flag is None else float(row.occupied_flag) for row in rows],
        dtype=float,
    )
    room_temperature = np.asarray(
        [
            np.nan if row.room_temperature_c is None else float(row.room_temperature_c)
            for row in rows
        ],
        dtype=float,
    )

    outdoor_temperature = _sanitize_nonfinite(outdoor_temperature, fallback=np.nan)
    thermal_output = _sanitize_nonfinite(thermal_output, fallback=0.0)
    solar_gain = _sanitize_nonfinite(solar_gain, fallback=0.0)
    solar_irradiance = _sanitize_nonfinite(solar_irradiance, fallback=0.0)
    shutter_position = _sanitize_nonfinite(shutter_position, fallback=0.0)
    occupied_flag = _sanitize_nonfinite(occupied_flag, fallback=0.0)
    room_temperature = _sanitize_nonfinite(room_temperature, fallback=np.nan)

    solar_shutter_interaction = solar_irradiance * np.clip(shutter_position, 0.0, 100.0) / 100.0

    is_sunny = solar_irradiance >= config.sunny_irradiance_threshold_w_m2
    is_heating_active = thermal_output >= config.heating_active_threshold_kw
    is_shutters_open = shutter_position >= config.shutters_open_min_pct
    is_shutters_closed = shutter_position <= config.shutters_closed_max_pct
    is_night = (local_hours >= config.night_start_hour) | (local_hours < config.night_end_hour)
    is_freezing = outdoor_temperature <= config.freezing_outdoor_temperature_max_c

    return PreparedRoomData(
        timestamps_utc=timestamps_utc,
        field_arrays={
            "room_temperature_c": room_temperature,
            "outdoor_temperature_c": outdoor_temperature,
            "thermal_output_estimate_kw": thermal_output,
            "solar_gain_proxy_w_m2": solar_gain,
            "solar_irradiance_w_m2": solar_irradiance,
            "shutter_position_pct": shutter_position,
            "solar_shutter_interaction": solar_shutter_interaction,
            "occupied_flag": occupied_flag,
        },
        segment_masks={
            "sunny": is_sunny,
            "heating_active": is_heating_active,
            "cold_weather": outdoor_temperature <= config.cold_outdoor_temperature_max_c,
            "freezing_weather": is_freezing,
            "mild_weather": outdoor_temperature >= config.mild_outdoor_temperature_min_c,
            "freezing_and_heating": is_freezing & is_heating_active,
            "freezing_night": is_freezing & is_night,
            "shutters_open": is_shutters_open,
            "shutters_closed": is_shutters_closed,
            "sunny_shutters_open": is_sunny & is_shutters_open,
            "sunny_shutters_closed": is_sunny & is_shutters_closed,
            "heating_and_sunny": is_sunny & is_heating_active,
            "night": is_night,
            "occupied": occupied_flag >= 1.0,
            "sunny_midday": is_sunny
            & (local_hours >= config.sunny_midday_start_hour)
            & (local_hours < config.sunny_midday_end_hour),
        },
    )


def segment_definitions(config) -> list[tuple[str, str]]:
    return [
        ("sunny", f"solar irradiance >= {config.sunny_irradiance_threshold_w_m2:.0f} W/m2"),
        ("heating_active", f"thermal output >= {config.heating_active_threshold_kw:.2f} kW"),
        ("cold_weather", f"outdoor temperature <= {config.cold_outdoor_temperature_max_c:.1f} C"),
        (
            "freezing_weather",
            f"outdoor temperature <= {config.freezing_outdoor_temperature_max_c:.1f} C",
        ),
        ("mild_weather", f"outdoor temperature >= {config.mild_outdoor_temperature_min_c:.1f} C"),
        ("freezing_and_heating", "freezing weather with heating active"),
        ("freezing_night", "freezing weather during night hours"),
        ("shutters_open", f"shutter position >= {config.shutters_open_min_pct:.0f}%"),
        ("shutters_closed", f"shutter position <= {config.shutters_closed_max_pct:.0f}%"),
        ("sunny_shutters_open", "sunny with shutters open"),
        ("sunny_shutters_closed", "sunny with shutters closed"),
        ("heating_and_sunny", "heating active with sunny conditions"),
        (
            "night",
            (
                f"local hour in "
                f"[{config.night_start_hour:02d}:00, 24:00) or [00:00, {config.night_end_hour:02d}:00)"
            ),
        ),
        ("occupied", "occupied_flag == 1"),
        (
            "sunny_midday",
            (
                "sunny and local hour in "
                f"[{config.sunny_midday_start_hour:02d}:00, {config.sunny_midday_end_hour:02d}:00)"
            ),
        ),
    ]


def row_segments(row: MpcDatasetRow, config) -> set[str]:
    segments: set[str] = set()
    solar_irradiance = row.solar_irradiance_w_m2 or 0.0
    thermal_output = row.thermal_output_estimate_kw or 0.0
    outdoor_temperature = row.outdoor_temperature_c
    shutter_position = row.shutter_position_pct
    local_hour = row.timestamp_utc.astimezone().hour
    occupied_flag = row.occupied_flag

    is_sunny = solar_irradiance >= config.sunny_irradiance_threshold_w_m2
    is_heating_active = thermal_output >= config.heating_active_threshold_kw
    is_shutters_open = (
        shutter_position is not None and shutter_position >= config.shutters_open_min_pct
    )
    is_shutters_closed = (
        shutter_position is not None and shutter_position <= config.shutters_closed_max_pct
    )
    is_night = local_hour >= config.night_start_hour or local_hour < config.night_end_hour
    is_freezing = (
        outdoor_temperature is not None
        and outdoor_temperature <= config.freezing_outdoor_temperature_max_c
    )

    if outdoor_temperature is not None and outdoor_temperature <= config.cold_outdoor_temperature_max_c:
        segments.add("cold_weather")
    if is_freezing:
        segments.add("freezing_weather")
    if outdoor_temperature is not None and outdoor_temperature >= config.mild_outdoor_temperature_min_c:
        segments.add("mild_weather")
    if is_sunny:
        segments.add("sunny")
        if config.sunny_midday_start_hour <= local_hour < config.sunny_midday_end_hour:
            segments.add("sunny_midday")
    if is_heating_active:
        segments.add("heating_active")
    if is_shutters_open:
        segments.add("shutters_open")
    if is_shutters_closed:
        segments.add("shutters_closed")
    if is_sunny and is_shutters_open:
        segments.add("sunny_shutters_open")
    if is_sunny and is_shutters_closed:
        segments.add("sunny_shutters_closed")
    if is_sunny and is_heating_active:
        segments.add("heating_and_sunny")
    if is_night:
        segments.add("night")
    if is_freezing and is_heating_active:
        segments.add("freezing_and_heating")
    if is_freezing and is_night:
        segments.add("freezing_night")
    if occupied_flag:
        segments.add("occupied")

    return segments
