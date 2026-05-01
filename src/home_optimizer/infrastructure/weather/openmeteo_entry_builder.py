from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.names import GTI_LIVING_ROOM_WINDOWS, GTI_PV
from home_optimizer.domain.time import ensure_utc

from .ports import ForecastRepositoryPort, OpenMeteoGatewayPort

BASE_VARIABLES = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "wind": "wind_speed_10m",
    "dew_point": "dew_point_2m",
    "direct_radiation": "direct_radiation",
    "diffuse_radiation": "diffuse_radiation",
}
GTI_VARIABLE = "global_tilted_irradiance"
FORECAST_UNITS = {
    "temperature": "degC",
    "humidity": "%",
    "wind": "ms",
    "dew_point": "degC",
    "direct_radiation": "Wm2",
    "diffuse_radiation": "Wm2",
    GTI_PV: "Wm2",
    GTI_LIVING_ROOM_WINDOWS: "Wm2",
}
LIVING_ROOM_WINDOW_TILT = 90.0


class OpenMeteoForecastEntryBuilder:
    def __init__(
        self,
        gateway: OpenMeteoGatewayPort,
        repository: ForecastRepositoryPort,
        *,
        pv_tilt: float | None,
        pv_azimuth: float | None,
        living_room_window_azimuth: float | None,
        forecast_steps: int,
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.pv_tilt = pv_tilt
        self.pv_azimuth = pv_azimuth
        self.living_room_window_azimuth = living_room_window_azimuth
        self.forecast_steps = forecast_steps

    def build_entries(
        self,
        *,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        entries = self._build_base_entries(
            fetched_at,
            latitude,
            longitude,
            past_days=past_days,
            use_forecast_time_as_created_at=use_forecast_time_as_created_at,
        )
        entries.extend(
            self._build_gti_entries(
                fetched_at,
                latitude,
                longitude,
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
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        payload = self.gateway.fetch_minutely_forecast(
            latitude=latitude,
            longitude=longitude,
            variables=list(BASE_VARIABLES.values()),
            forecast_steps=self.forecast_steps,
            past_days=past_days,
        )
        return self._entries_from_payload(
            payload=payload,
            fetched_at=fetched_at,
            variable_map=BASE_VARIABLES,
            use_forecast_time_as_created_at=use_forecast_time_as_created_at,
        )

    def _build_gti_entries(
        self,
        fetched_at: datetime,
        latitude: float,
        longitude: float,
        *,
        past_days: int | None = None,
        use_forecast_time_as_created_at: bool = False,
    ) -> list[ForecastEntry]:
        entries: list[ForecastEntry] = []

        if self.pv_tilt is not None and self.pv_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=self.forecast_steps,
                past_days=past_days,
                tilt=self.pv_tilt,
                azimuth=_compass_to_open_meteo_azimuth(self.pv_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload,
                    fetched_at=fetched_at,
                    variable_map={GTI_PV: GTI_VARIABLE},
                    use_forecast_time_as_created_at=use_forecast_time_as_created_at,
                )
            )

        if self.living_room_window_azimuth is not None:
            payload = self.gateway.fetch_minutely_forecast(
                latitude=latitude,
                longitude=longitude,
                variables=[GTI_VARIABLE],
                forecast_steps=self.forecast_steps,
                past_days=past_days,
                tilt=LIVING_ROOM_WINDOW_TILT,
                azimuth=_compass_to_open_meteo_azimuth(self.living_room_window_azimuth),
            )
            entries.extend(
                self._entries_from_payload(
                    payload=payload,
                    fetched_at=fetched_at,
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
                        created_at_utc=forecast_time if use_forecast_time_as_created_at else fetched_at,
                        forecast_time_utc=forecast_time,
                        name=name,
                        value=float(value),
                        unit=FORECAST_UNITS.get(name),
                        source=self.repository.source,
                    )
                )

        return entries


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
