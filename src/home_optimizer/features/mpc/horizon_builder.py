from __future__ import annotations

from datetime import timedelta

from home_optimizer.domain.names import (
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
)
from home_optimizer.domain.series_transforms import build_daily_target_band_series
from home_optimizer.domain.time import ensure_utc, parse_datetime
from home_optimizer.features.mpc.exogenous_features import (
    continue_exp_filter,
    local_hour_sin_cos,
)
from home_optimizer.features.mpc.models import MpcHorizonBuildRequest, MpcHorizonStep


class MpcHorizonBuilder:
    def build(self, request: MpcHorizonBuildRequest) -> list[MpcHorizonStep]:
        start_time_utc = ensure_utc(request.start_time_utc)
        interval = timedelta(minutes=request.interval_minutes)
        end_time_utc = start_time_utc + (interval * request.horizon_steps)
        target_series, min_series, max_series = build_daily_target_band_series(
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
        target_by_timestamp = {
            parse_datetime(point.timestamp): point.value for point in target_series.points
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
        solar_direct_by_timestamp = {
            timestamp_utc: float(value * request.solar_gain_input_scale)
            for timestamp_utc, value in solar_by_timestamp.items()
        }
        ordered_timestamps = [
            start_time_utc + (interval * step_index)
            for step_index in range(request.horizon_steps)
        ]
        solar_direct_values = [
            float(solar_direct_by_timestamp.get(timestamp_utc, 0.0))
            for timestamp_utc in ordered_timestamps
        ]
        solar_filtered_values = continue_exp_filter(
            solar_direct_values,
            alpha=request.solar_gain_filter_alpha,
            initial_filtered_value=request.initial_filtered_solar_gain_kw,
        )

        horizon: list[MpcHorizonStep] = []
        for step_index in range(request.horizon_steps):
            timestamp_utc = ordered_timestamps[step_index]
            temp_min_c = min_by_timestamp.get(timestamp_utc, request.fallback_temp_min_c)
            temp_max_c = max_by_timestamp.get(timestamp_utc, request.fallback_temp_max_c)
            if temp_min_c is None or temp_max_c is None:
                raise ValueError(
                    "Missing target temperature bounds for MPC horizon step "
                    f"at {timestamp_utc.isoformat()}"
                )
            hour_sin, hour_cos = local_hour_sin_cos(
                timestamp_utc,
                local_timezone=request.local_timezone,
            )
            horizon.append(
                MpcHorizonStep(
                    timestamp_utc=timestamp_utc,
                    outdoor_temp_c=float(outdoor_by_timestamp.get(timestamp_utc, 0.0)),
                    solar_gain_kw=solar_direct_values[step_index],
                    solar_gain_mass_kw=solar_filtered_values[step_index],
                    solar_irradiance_forecast_w_m2=float(
                        solar_by_timestamp.get(timestamp_utc, 0.0)
                    ),
                    effective_heating_kw_forecast=request.default_effective_heating_kw,
                    hp_electric_power_forecast_kw=request.default_hp_electric_power_kw,
                    pv_available_power_forecast_kw=float(
                        pv_by_timestamp.get(timestamp_utc, 0.0) * request.pv_power_input_scale
                    ),
                    base_load_power_forecast_kw=request.default_base_load_power_kw,
                    occupied=request.default_occupied,
                    hour_sin=hour_sin,
                    hour_cos=hour_cos,
                    target_temp_c=float(
                        target_by_timestamp.get(
                            timestamp_utc,
                            (float(temp_min_c) + float(temp_max_c)) / 2.0,
                        )
                    ),
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
