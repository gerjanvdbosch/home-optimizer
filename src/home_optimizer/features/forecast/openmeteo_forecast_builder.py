from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.names import (
    FORECAST_DEW_POINT,
    FORECAST_DIFFUSE_RADIATION,
    FORECAST_DIRECT_RADIATION,
    FORECAST_HUMIDITY,
    FORECAST_PRECIPITATION,
    FORECAST_TEMPERATURE,
    FORECAST_WEATHER_CODE,
    FORECAST_WIND,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_PV,
)
from home_optimizer.domain.time import ensure_utc

from .ports import ForecastRepositoryPort, OpenMeteoGatewayPort

GTI_VARIABLE = "global_tilted_irradiance"
LIVING_ROOM_WINDOW_TILT = 90.0

OPEN_METEO_BASE_VARIABLES = {
    FORECAST_TEMPERATURE: "temperature_2m",
    FORECAST_HUMIDITY: "relative_humidity_2m",
    FORECAST_WIND: "wind_speed_10m",
    FORECAST_DEW_POINT: "dew_point_2m",
    FORECAST_PRECIPITATION: "precipitation",
    FORECAST_WEATHER_CODE: "weather_code",
    FORECAST_DIRECT_RADIATION: "direct_radiation",
    FORECAST_DIFFUSE_RADIATION: "diffuse_radiation",
}
OPEN_METEO_WEATHER_UNITS = {
    FORECAST_TEMPERATURE: "°C",
    FORECAST_HUMIDITY: "%",
    FORECAST_WIND: "ms",
    FORECAST_DEW_POINT: "°C",
    FORECAST_PRECIPITATION: "mm",
    FORECAST_WEATHER_CODE: "code",
    FORECAST_DIRECT_RADIATION: "W/m2",
    FORECAST_DIFFUSE_RADIATION: "W/m2",
    GTI_PV: "W/m2",
    GTI_LIVING_ROOM_WINDOWS: "W/m2",
}

TEntry = TypeVar("TEntry")


class OpenMeteoForecastEntryBuilder:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        repository: ForecastRepositoryPort,
        *,
        pv_tilt: float | None,
        pv_azimuth: float | None,
        living_room_window_azimuth: float | None,
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.pv_tilt = pv_tilt
        self.pv_azimuth = pv_azimuth
        self.living_room_window_azimuth = living_room_window_azimuth

    def build_entries(
        self,
        *,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
        forecast_steps: int | None = None,
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        entries = self._build_base_entries(
            fetched_at, latitude, longitude,
            forecast_steps=forecast_steps, past_days=past_days,
            use_forecast_time_as_created_at=use_forecast_time_as_created_at,
        )
        entries.extend(
            self._build_gti_entries(
                fetched_at, latitude, longitude,
                forecast_steps=forecast_steps, past_days=past_days,
                use_forecast_time_as_created_at=use_forecast_time_as_created_at,
            )
        )
        return entries

    def _build_base_entries(
        self,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
        *,
        forecast_steps: int | None = None,
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        payload = self.gateway.fetch_minutely_forecast(
            latitude=latitude,
            longitude=longitude,
            variables=list(OPEN_METEO_BASE_VARIABLES.values()),
            forecast_steps=forecast_steps,
            past_days=past_days,
        )
        return self._entries_from_payload(
            payload=payload,
            fetched_at=fetched_at,
            variable_map=OPEN_METEO_BASE_VARIABLES,
            use_forecast_time_as_created_at=use_forecast_time_as_created_at,
        )

    def _build_gti_entries(
        self,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
        *,
        forecast_steps: int | None = None,
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        entries: list[ForecastEntry] = []

        if self.pv_tilt is not None and self.pv_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=forecast_steps,
                past_days=past_days,
                tilt=self.pv_tilt,
                azimuth=_compass_to_open_meteo_azimuth(self.pv_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload, fetched_at=fetched_at,
                    variable_map={GTI_PV: GTI_VARIABLE},
                    use_forecast_time_as_created_at=use_forecast_time_as_created_at,
                )
            )

        if self.living_room_window_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=forecast_steps,
                past_days=past_days,
                tilt=LIVING_ROOM_WINDOW_TILT,
                azimuth=_compass_to_open_meteo_azimuth(self.living_room_window_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload, fetched_at=fetched_at,
                    variable_map={GTI_LIVING_ROOM_WINDOWS: GTI_VARIABLE},
                    use_forecast_time_as_created_at=use_forecast_time_as_created_at,
                )
            )

        return entries

    def _entries_from_payload(
        self,
        *,
        payload: dict[str, object],
        fetched_at: datetime,
        variable_map: dict[str, str],
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        return _build_entries_from_payload(
            payload=payload,
            section_name="minutely_15",
            variable_map=variable_map,
            parse_timestamp=_parse_minutely_open_meteo_timestamp,
            entry_factory=lambda forecast_time, name, value: ForecastEntry(
                created_at_utc=forecast_time if use_forecast_time_as_created_at else fetched_at,
                forecast_time_utc=forecast_time,
                name=name,
                value=value,
                unit=OPEN_METEO_WEATHER_UNITS.get(name),
                source=self.repository.source,
            ),
            missing_payload_message="Open-Meteo response is missing minutely_15 data",
            missing_timestamp_message="Open-Meteo response is missing forecast timestamps",
            missing_variable_message="Open-Meteo response is missing {variable}",
            length_mismatch_message="Open-Meteo response length mismatch for {variable}",
        )


def _build_entries_from_payload(
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


def _parse_minutely_open_meteo_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo timestamp must be a string")
    return ensure_utc(datetime.fromisoformat(f"{value}:00+00:00"))


def _compass_to_open_meteo_azimuth(compass_degrees: float) -> float:
    converted = compass_degrees - 180.0
    if converted > 180.0:
        converted -= 360.0
    if converted <= -180.0:
        converted += 360.0
    return converted

