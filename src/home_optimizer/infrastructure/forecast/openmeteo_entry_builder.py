from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.forecast import ForecastEntry

from .openmeteo_common import (
    GTI_VARIABLE,
    OPEN_METEO_BASE_VARIABLES,
    OPEN_METEO_WEATHER_UNITS,
    build_entries_from_payload,
    build_gti_entries,
    parse_minutely_open_meteo_timestamp,
)
from .ports import ForecastRepositoryPort, OpenMeteoGatewayPort


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
            fetched_at,
            latitude,
            longitude,
            forecast_steps=forecast_steps,
            past_days=past_days,
            use_forecast_time_as_created_at=use_forecast_time_as_created_at,
        )
        entries.extend(
            self._build_gti_entries(
                fetched_at,
                latitude,
                longitude,
                forecast_steps=forecast_steps,
                past_days=past_days,
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
        return build_gti_entries(
            pv_tilt=self.pv_tilt,
            pv_azimuth=self.pv_azimuth,
            living_room_window_azimuth=self.living_room_window_azimuth,
            fetch_payload=lambda tilt, azimuth: self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=forecast_steps,
                past_days=past_days,
                tilt=tilt,
                azimuth=azimuth,
            ),
            entries_from_payload=lambda payload, variable_map: self._entries_from_payload(
                payload=payload,
                fetched_at=fetched_at,
                variable_map=variable_map,
                use_forecast_time_as_created_at=use_forecast_time_as_created_at,
            ),
        )

    def _entries_from_payload(
        self,
        *,
        payload: dict[str, object],
        fetched_at: datetime,
        variable_map: dict[str, str],
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        return build_entries_from_payload(
            payload=payload,
            section_name="minutely_15",
            variable_map=variable_map,
            parse_timestamp=parse_minutely_open_meteo_timestamp,
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
