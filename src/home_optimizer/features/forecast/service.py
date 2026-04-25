from __future__ import annotations

import logging
from datetime import datetime

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.time import ensure_utc

from .ports import ForecastRepositoryPort, HomeLocationGatewayPort, OpenMeteoGatewayPort

LOGGER = logging.getLogger(__name__)

BASE_VARIABLES = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "wind": "wind_speed_10m",
    "dew_point": "dew_point_2m",
    "direct_radiation": "direct_radiation",
    "diffuse_radiation": "diffuse_radiation",
}
GTI_VARIABLE = "global_tilted_irradiance"
PV_GTI_NAME = "gti_pv"
LIVING_ROOM_GTI_NAME = "gti_living_room_windows"
FORECAST_UNITS = {
    "temperature": "degC",
    "humidity": "%",
    "wind": "ms",
    "dew_point": "degC",
    "direct_radiation": "Wm2",
    "diffuse_radiation": "Wm2",
    PV_GTI_NAME: "Wm2",
    LIVING_ROOM_GTI_NAME: "Wm2",
}
HOME_ZONE_ENTITY_ID = "zone.home"
LIVING_ROOM_WINDOW_TILT = 90.0


class OpenMeteoForecastService:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        home_location_gateway: HomeLocationGatewayPort,
        repository: ForecastRepositoryPort,
        *,
        enabled: bool,
        pv_tilt: float | None,
        pv_azimuth: float | None,
        living_room_window_azimuth: float | None,
        forecast_steps: int = 192,
    ) -> None:
        self.gateway = gateway
        self.home_location_gateway = home_location_gateway
        self.repository = repository
        self._enabled = enabled
        self.pv_tilt = pv_tilt
        self.pv_azimuth = pv_azimuth
        self.living_room_window_azimuth = living_room_window_azimuth
        self.forecast_steps = forecast_steps

    @property
    def enabled(self) -> bool:
        return self._enabled

    def refresh_forecast(self, created_at: datetime | None = None) -> int:
        if not self.enabled:
            LOGGER.info("Open-Meteo forecast refresh skipped: disabled")
            return 0

        fetched_at = ensure_utc(created_at or utc_now())
        coordinates = self._get_home_coordinates()
        if coordinates is None:
            LOGGER.info("Open-Meteo forecast refresh skipped: zone.home coordinates unavailable")
            return 0

        latitude, longitude = coordinates

        entries = self._build_base_entries(fetched_at, latitude, longitude)
        entries.extend(self._build_gti_entries(fetched_at, latitude, longitude))
        self.repository.write_entries(entries)
        LOGGER.info("Stored %s Open-Meteo forecast values", len(entries))
        return len(entries)

    def _build_base_entries(
        self,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
    ) -> list[ForecastEntry]:
        payload = self.gateway.fetch_minutely_forecast(
            latitude=latitude,
            longitude=longitude,
            variables=list(BASE_VARIABLES.values()),
            forecast_steps=self.forecast_steps,
        )
        return self._entries_from_payload(
            payload=payload,
            fetched_at=fetched_at,
            variable_map=BASE_VARIABLES,
        )

    def _build_gti_entries(
        self,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
    ) -> list[ForecastEntry]:
        entries: list[ForecastEntry] = []

        if self.pv_tilt is not None and self.pv_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=self.forecast_steps,
                tilt=self.pv_tilt,
                azimuth=_compass_to_open_meteo_azimuth(self.pv_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload,
                    fetched_at=fetched_at,
                    variable_map={PV_GTI_NAME: GTI_VARIABLE},
                )
            )

        if self.living_room_window_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=self.forecast_steps,
                tilt=LIVING_ROOM_WINDOW_TILT,
                azimuth=_compass_to_open_meteo_azimuth(self.living_room_window_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload,
                    fetched_at=fetched_at,
                    variable_map={LIVING_ROOM_GTI_NAME: GTI_VARIABLE},
                )
            )

        return entries

    def _entries_from_payload(
        self,
        *,
        payload: dict[str, object],
        fetched_at: datetime,
        variable_map: dict[str, str],
    ) -> list[ForecastEntry]:
        minutely = payload.get("minutely_15")
        if not isinstance(minutely, dict):
            raise ValueError("Open-Meteo response is missing minutely_15 data")

        raw_times = minutely.get("time")
        if not isinstance(raw_times, list):
            raise ValueError("Open-Meteo response is missing forecast timestamps")

        times = [_parse_open_meteo_timestamp(value) for value in raw_times]
        entries: list[ForecastEntry] = []

        for name, variable in variable_map.items():
            raw_values = minutely.get(variable)
            if not isinstance(raw_values, list):
                raise ValueError(f"Open-Meteo response is missing {variable}")
            if len(raw_values) != len(times):
                raise ValueError(f"Open-Meteo response length mismatch for {variable}")

            for forecast_time, value in zip(times, raw_values, strict=True):
                if value is None:
                    continue
                entries.append(
                    ForecastEntry(
                        created_at_utc=fetched_at,
                        forecast_time_utc=forecast_time,
                        name=name,
                        value=float(value),
                        unit=FORECAST_UNITS.get(name),
                        source=self.repository.source,
                    )
                )

        return entries

    def _get_home_coordinates(self) -> tuple[float, float] | None:
        state = self.home_location_gateway.get_state(HOME_ZONE_ENTITY_ID)
        attributes = state.get("attributes")
        if not isinstance(attributes, dict):
            return None

        latitude = _parse_coordinate(attributes.get("latitude"))
        longitude = _parse_coordinate(attributes.get("longitude"))
        if latitude is None or longitude is None:
            return None

        return latitude, longitude


def _parse_open_meteo_timestamp(value: object) -> datetime:
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


def _parse_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
