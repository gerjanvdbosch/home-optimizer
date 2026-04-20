"""History-based DHW tap-flow forecaster.

This provider estimates the exogenous tap-flow disturbance ``V_tap[k]`` from
persisted telemetry using the DHW total-energy balance and then projects that
history onto a recurring hour-of-day profile. The result is a horizon-wide
``dhw_v_tap_forecast`` array that the runtime MPC can use as a known LTV
parameter instead of assuming a constant draw all day long.

Units: power [kW], energy [kWh], temperature [°C], volume flow [m³/h], time [h].
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..telemetry.models import TelemetryAggregate

_SECONDS_PER_HOUR: float = 3600.0
_DHW_MODE_NAME: str = "dhw"


@dataclass(frozen=True, slots=True)
class DHWTapForecastSettings:
    """Fail-fast settings for the history-based DHW tap forecaster.

    Attributes:
        min_training_samples: Minimum number of valid historical transition samples
            required before an hourly tap profile is trusted [-].
        min_samples_per_hour: Minimum number of historical samples required for one
            hour-of-day bucket before that bucket is used directly [-]. Sparse
            hours fall back to zero expected draw instead of inventing demand.
        max_history_gap_hours: Maximum allowed time gap between two telemetry
            buckets that still counts as one continuous transition [h].
        max_implied_tap_m3_per_h: Hard upper clamp on implied tap flow [m³/h].
            This prevents one noisy bucket from dominating the forecast profile.
    """

    min_training_samples: int = 24
    min_samples_per_hour: int = 2
    max_history_gap_hours: float = 2.0
    max_implied_tap_m3_per_h: float = 0.2

    def __post_init__(self) -> None:
        if self.min_training_samples <= 0:
            raise ValueError("min_training_samples must be strictly positive.")
        if self.min_samples_per_hour <= 0:
            raise ValueError("min_samples_per_hour must be strictly positive.")
        if self.max_history_gap_hours <= 0.0:
            raise ValueError("max_history_gap_hours must be strictly positive.")
        if self.max_implied_tap_m3_per_h <= 0.0:
            raise ValueError("max_implied_tap_m3_per_h must be strictly positive.")


class DHWTapForecaster:
    """Forecast recurring DHW tap flow from persisted telemetry history.

    The forecaster does **not** fit or persist an ML artifact. Instead it derives a
    physically interpretable hour-of-day draw profile directly from historical tank
    energy balances. This keeps the implementation transparent and robust for small
    datasets while still letting the MPC anticipate recurring evening showers.
    """

    def __init__(self, settings: DHWTapForecastSettings | None = None) -> None:
        self._settings = settings or DHWTapForecastSettings()

    @staticmethod
    def _total_energy_kwh(*, t_top_c: float, t_bot_c: float, c_top_kwh_per_k: float, c_bot_kwh_per_k: float) -> float:
        """Return total tank energy proxy ``C_top·T_top + C_bot·T_bot`` [kWh]."""
        return c_top_kwh_per_k * t_top_c + c_bot_kwh_per_k * t_bot_c

    def _infer_v_tap_m3_per_h(
        self,
        *,
        previous_row: TelemetryAggregate,
        next_row: TelemetryAggregate,
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
    ) -> float | None:
        """Infer one tap-flow sample from the DHW total-energy balance.

        Implements the rearranged tank balance from §9.5:

            V_tap = (P_dhw - Q_loss - ΔE/Δt) / (λ·(T_top - T_mains))

        using the mean outlet/ambient temperatures over the interval and clamping the
        result to the physically feasible interval ``[0, max_implied_tap_m3_per_h]``.
        """
        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > self._settings.max_history_gap_hours:
            return None

        start_energy_kwh = self._total_energy_kwh(
            t_top_c=float(previous_row.dhw_top_temperature_last_c),
            t_bot_c=float(previous_row.dhw_bottom_temperature_last_c),
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
        )
        end_energy_kwh = self._total_energy_kwh(
            t_top_c=float(next_row.dhw_top_temperature_last_c),
            t_bot_c=float(next_row.dhw_bottom_temperature_last_c),
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
        )
        delta_energy_rate_kw = (end_energy_kwh - start_energy_kwh) / dt_hours
        mean_t_top_c = 0.5 * (
            float(previous_row.dhw_top_temperature_last_c) + float(next_row.dhw_top_temperature_last_c)
        )
        mean_t_bot_c = 0.5 * (
            float(previous_row.dhw_bottom_temperature_last_c) + float(next_row.dhw_bottom_temperature_last_c)
        )
        t_amb_c = float(next_row.boiler_ambient_temp_mean_c)
        t_mains_c = float(next_row.t_mains_estimated_mean_c)
        q_loss_kw = (mean_t_top_c - t_amb_c) / r_loss_k_per_kw + (mean_t_bot_c - t_amb_c) / r_loss_k_per_kw
        p_dhw_kw = (
            max(float(next_row.hp_thermal_power_mean_kw), 0.0)
            if str(next_row.hp_mode_last).strip().lower() == _DHW_MODE_NAME
            else 0.0
        )

        tap_denominator = lambda_water_kwh_per_m3_k * (mean_t_top_c - t_mains_c)
        if tap_denominator <= 0.0:
            return 0.0

        implied_v_tap = (p_dhw_kw - q_loss_kw - delta_energy_rate_kw) / tap_denominator
        clamped_v_tap = float(np.clip(implied_v_tap, 0.0, self._settings.max_implied_tap_m3_per_h))
        return clamped_v_tap

    def predict_from_repository(
        self,
        *,
        repository: Any,
        horizon_valid_at_utc: Sequence[datetime],
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
    ) -> np.ndarray | None:
        """Return an hour-of-day DHW tap-flow forecast for the requested horizon.

        Args:
            repository: Telemetry repository exposing ``list_aggregates()``.
            horizon_valid_at_utc: Forecast validity timestamps for the upcoming MPC
                horizon [UTC].
            c_top_kwh_per_k: DHW top-layer heat capacity ``C_top`` [kWh/K].
            c_bot_kwh_per_k: DHW bottom-layer heat capacity ``C_bot`` [kWh/K].
            r_loss_k_per_kw: DHW standby-loss resistance ``R_loss`` [K/kW].
            lambda_water_kwh_per_m3_k: Water heat-capacity constant ``λ`` [kWh/(m³·K)].

        Returns:
            1-D NumPy array with one ``V_tap`` estimate per requested timestamp, or
            ``None`` when insufficient history is available.
        """
        if not horizon_valid_at_utc:
            raise ValueError("horizon_valid_at_utc must not be empty.")
        rows = sorted(repository.list_aggregates(), key=lambda row: row.bucket_end_utc)
        if len(rows) < 2:
            return None

        samples_by_hour: dict[int, list[float]] = defaultdict(list)
        valid_sample_count = 0
        for previous_row, next_row in zip(rows, rows[1:]):
            inferred_v_tap = self._infer_v_tap_m3_per_h(
                previous_row=previous_row,
                next_row=next_row,
                c_top_kwh_per_k=c_top_kwh_per_k,
                c_bot_kwh_per_k=c_bot_kwh_per_k,
                r_loss_k_per_kw=r_loss_k_per_kw,
                lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
            )
            if inferred_v_tap is None:
                continue
            samples_by_hour[int(next_row.bucket_end_utc.hour)].append(inferred_v_tap)
            valid_sample_count += 1

        if valid_sample_count < self._settings.min_training_samples:
            return None

        forecast_values: list[float] = []
        for valid_at_utc in horizon_valid_at_utc:
            hour_samples = samples_by_hour[int(valid_at_utc.hour)]
            if len(hour_samples) < self._settings.min_samples_per_hour:
                forecast_values.append(0.0)
                continue
            forecast_values.append(float(np.mean(hour_samples)))
        return np.asarray(forecast_values, dtype=float)

