from __future__ import annotations

from datetime import timedelta

from home_optimizer.domain.names import (
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
)
from home_optimizer.domain.series_transforms import build_daily_target_band_series
from home_optimizer.domain.time import ensure_utc, parse_datetime
from home_optimizer.features.mpc.models import MpcHorizonBuildRequest, MpcHorizonStep


class MpcHorizonBuilder:
    def build(self, request: MpcHorizonBuildRequest) -> list[MpcHorizonStep]:
        start_time_utc = ensure_utc(request.start_time_utc)
        interval = timedelta(minutes=request.interval_minutes)
        end_time_utc = start_time_utc + (interval * request.horizon_steps)
        _, min_series, max_series = build_daily_target_band_series(
            request.target_schedule,
            start_time=start_time_utc,
            end_time=end_time_utc,
            target_name="room_target_temperature",
            minimum_name="room_target_min_temperature",
            maximum_name="room_target_max_temperature",
            interval_minutes=request.interval_minutes,
        )
        min_by_timestamp = {
            parse_datetime(point.timestamp): point.value for point in min_series.points
        }
        max_by_timestamp = {
            parse_datetime(point.timestamp): point.value for point in max_series.points
        }
        outdoor_by_timestamp = self._forecast_values_by_timestamp(
            request,
            request.outdoor_temperature_name,
        )
        solar_by_timestamp = self._forecast_values_by_timestamp(
            request,
            request.solar_gain_name,
        )
        pv_by_timestamp = self._forecast_values_by_timestamp(
            request,
            request.pv_power_name,
        )
        price_by_timestamp = self._price_values_by_timestamp(request)

        horizon: list[MpcHorizonStep] = []
        for step_index in range(request.horizon_steps):
            timestamp_utc = start_time_utc + (interval * step_index)
            temp_min_c = min_by_timestamp.get(timestamp_utc, request.fallback_temp_min_c)
            temp_max_c = max_by_timestamp.get(timestamp_utc, request.fallback_temp_max_c)
            if temp_min_c is None or temp_max_c is None:
                raise ValueError(
                    "Missing target temperature bounds for MPC horizon step "
                    f"at {timestamp_utc.isoformat()}"
                )
            horizon.append(
                MpcHorizonStep(
                    timestamp_utc=timestamp_utc,
                    outdoor_temp_c=float(outdoor_by_timestamp.get(timestamp_utc, 0.0)),
                    solar_gain_kw=float(
                        solar_by_timestamp.get(timestamp_utc, 0.0) * request.solar_gain_input_scale
                    ),
                    effective_heating_kw_forecast=request.default_effective_heating_kw,
                    hp_electric_power_forecast_kw=request.default_hp_electric_power_kw,
                    pv_available_power_forecast_kw=float(
                        request.pv_power_forecast_by_timestamp.get(
                            timestamp_utc,
                            pv_by_timestamp.get(timestamp_utc, 0.0)
                            * request.pv_power_input_scale,
                        )
                    ),
                    base_load_power_forecast_kw=float(
                        request.base_load_power_forecast_by_timestamp.get(
                            timestamp_utc,
                            request.default_base_load_power_kw,
                        )
                    ),
                    occupied=request.default_occupied,
                    temp_min_c=float(temp_min_c),
                    temp_max_c=float(temp_max_c),
                    price_eur_kwh=float(price_by_timestamp.get(timestamp_utc, 0.0)),
                    import_price_eur_kwh=float(price_by_timestamp.get(timestamp_utc, 0.0)),
                    export_price_eur_kwh=request.default_export_price_eur_kwh,
                )
            )
        return horizon

    def _forecast_values_by_timestamp(
        self,
        request: MpcHorizonBuildRequest,
        name: str,
    ) -> dict[object, float]:
        latest_by_forecast_time: dict[object, tuple[object, float]] = {}
        for entry in request.forecast_entries:
            if entry.name != name:
                continue
            forecast_time_utc = ensure_utc(entry.forecast_time_utc)
            created_at_utc = ensure_utc(entry.created_at_utc)
            existing = latest_by_forecast_time.get(forecast_time_utc)
            if existing is None or created_at_utc >= existing[0]:
                latest_by_forecast_time[forecast_time_utc] = (created_at_utc, float(entry.value))
        return {
            forecast_time_utc: value
            for forecast_time_utc, (_, value) in latest_by_forecast_time.items()
        }

    def _price_values_by_timestamp(self, request: MpcHorizonBuildRequest) -> dict[object, float]:
        price_by_timestamp: dict[object, float] = {}
        interval = timedelta(minutes=request.interval_minutes)
        for price_interval in request.price_intervals:
            cursor = ensure_utc(price_interval.start_time_utc)
            end_time_utc = ensure_utc(price_interval.end_time_utc)
            while cursor < end_time_utc:
                price_by_timestamp[cursor] = float(price_interval.value)
                cursor += interval
        return price_by_timestamp


DEFAULT_OUTDOOR_TEMPERATURE_FORECAST_NAME = FORECAST_TEMPERATURE
DEFAULT_SOLAR_GAIN_FORECAST_NAME = GTI_LIVING_ROOM_WINDOWS_ADJUSTED
