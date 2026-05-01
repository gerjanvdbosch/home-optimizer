from __future__ import annotations

from datetime import date

from home_optimizer.domain.historical_weather import HistoricalWeatherEntry

from .openmeteo_common import (
    GTI_VARIABLE,
    OPEN_METEO_BASE_VARIABLES,
    OPEN_METEO_WEATHER_UNITS,
    build_entries_from_payload,
    build_gti_entries,
    parse_hourly_open_meteo_timestamp,
)
from .ports import HistoricalWeatherRepositoryPort, OpenMeteoGatewayPort


class OpenMeteoHistoricalWeatherBuilder:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        repository: HistoricalWeatherRepositoryPort,
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
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalWeatherEntry]:
        entries = self._build_base_entries(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
        entries.extend(
            self._build_gti_entries(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
            )
        )
        return entries

    def _build_base_entries(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalWeatherEntry]:
        payload = self.gateway.fetch_hourly_historical_weather(
            latitude=latitude,
            longitude=longitude,
            variables=list(OPEN_METEO_BASE_VARIABLES.values()),
            start_date=start_date,
            end_date=end_date,
        )
        return self._entries_from_payload(
            payload=payload,
            variable_map=OPEN_METEO_BASE_VARIABLES,
        )

    def _build_gti_entries(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalWeatherEntry]:
        return build_gti_entries(
            pv_tilt=self.pv_tilt,
            pv_azimuth=self.pv_azimuth,
            living_room_window_azimuth=self.living_room_window_azimuth,
            fetch_payload=lambda tilt, azimuth: self.gateway.fetch_hourly_historical_weather(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                start_date=start_date,
                end_date=end_date,
                tilt=tilt,
                azimuth=azimuth,
            ),
            entries_from_payload=lambda payload, variable_map: self._entries_from_payload(
                payload=payload,
                variable_map=variable_map,
            ),
        )

    def _entries_from_payload(
        self,
        *,
        payload: dict[str, object],
        variable_map: dict[str, str],
    ) -> list[HistoricalWeatherEntry]:
        return build_entries_from_payload(
            payload=payload,
            section_name="hourly",
            variable_map=variable_map,
            parse_timestamp=parse_hourly_open_meteo_timestamp,
            entry_factory=lambda timestamp, name, value: HistoricalWeatherEntry(
                timestamp_utc=timestamp,
                name=name,
                value=value,
                unit=OPEN_METEO_WEATHER_UNITS.get(name),
                source=self.repository.source,
            ),
            missing_payload_message="Open-Meteo historical response is missing hourly data",
            missing_timestamp_message="Open-Meteo historical response is missing timestamps",
            missing_variable_message="Open-Meteo historical response is missing {variable}",
            length_mismatch_message="Open-Meteo historical response length mismatch for {variable}",
        )
