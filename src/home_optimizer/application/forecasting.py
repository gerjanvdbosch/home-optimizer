"""Forecast assembly helpers for the application-layer optimizer pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..domain.heat_pump.cop import HeatPumpCOPModel
from ..forecasting import ForecastService
from ..pricing import BasePriceModel, build_price_model
from ..types.constants import W_PER_KW
from ..types.forecast import DHWForecastHorizon, ForecastHorizon

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository
    from .optimizer import RunRequest

_FORECAST_SERVICE = ForecastService()


def inject_forecast_overrides(
    rows: list,
    n_steps: int,
    existing: dict | None = None,
) -> dict:
    """Build weather-forecast request overrides from persisted forecast rows."""
    existing = existing or {}
    slice_ = rows[:n_steps]
    overrides: dict = {}
    if "t_out_forecast" not in existing:
        overrides["t_out_forecast"] = [row.t_out_c for row in slice_]
    if "gti_window_forecast" not in existing:
        overrides["gti_window_forecast"] = [row.gti_w_per_m2 for row in slice_]
    if "gti_pv_forecast" not in existing:
        overrides["gti_pv_forecast"] = [row.gti_pv_w_per_m2 for row in slice_]
    return overrides


def build_repository_forecast_overrides(
    request: "RunRequest",
    repository: "TelemetryRepository",
    existing: dict | None = None,
) -> tuple[list, dict]:
    """Load persisted weather data and derived ML forecasts for one runtime request."""
    current_overrides = dict(existing or {})
    rows = repository.get_latest_forecast_batch()
    if not rows:
        return [], {}

    explicit_request_fields = request.model_dump(mode="python", exclude_none=True)
    forecast_overrides = inject_forecast_overrides(
        rows,
        request.horizon_hours,
        existing={**explicit_request_fields, **current_overrides},
    )
    current_overrides.update(forecast_overrides)
    materialized_request_data = {
        **request.model_dump(mode="python"),
        **current_overrides,
    }
    ml_overrides = _FORECAST_SERVICE.build_missing_overrides(
        request_data=materialized_request_data,
        repository=repository,
        weather_rows=rows[: request.horizon_hours],
        current_overrides=current_overrides,
    )
    forecast_overrides.update(ml_overrides)
    return rows, forecast_overrides


class ForecastBuilder:
    """Assemble fully materialised UFH and DHW forecast horizons from one request."""

    @staticmethod
    def materialize_horizon_array(
        *,
        name: str,
        horizon_steps: int,
        values: list[float] | None,
        fallback_scalar: float | None = None,
    ) -> np.ndarray:
        """Return a full-horizon forecast array with fail-fast length validation."""
        if values is None:
            if fallback_scalar is None:
                raise ValueError(f"{name} forecast is required for the full MPC horizon.")
            return np.full(horizon_steps, fallback_scalar, dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D forecast array.")
        if arr.size < horizon_steps:
            raise ValueError(
                f"{name} must provide at least {horizon_steps} values for the MPC horizon; "
                f"received {arr.size}."
            )
        return arr[:horizon_steps].copy()

    @staticmethod
    def build_ufh_forecast(
        req: "RunRequest",
        *,
        start_hour: int,
        cop_model: HeatPumpCOPModel,
    ) -> ForecastHorizon:
        """Build the UFH disturbance, price, PV, and COP forecast over the horizon."""
        forecast_config = req.ufh_forecast_config
        horizon_steps = forecast_config.horizon_steps
        price_model: BasePriceModel = build_price_model(forecast_config.price_config)
        prices = price_model.prices(start_hour, horizon_steps)
        feed_in_prices = price_model.feed_in_prices(start_hour, horizon_steps)

        t_out_arr = ForecastBuilder.materialize_horizon_array(
            name="t_out_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.t_out_forecast,
            fallback_scalar=forecast_config.outdoor_temperature_c,
        )
        gti_window = ForecastBuilder.materialize_horizon_array(
            name="gti_window_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.gti_window_forecast,
            fallback_scalar=0.0,
        )
        internal_gains = ForecastBuilder.build_internal_gains_profile(
            req,
            horizon_steps=horizon_steps,
        )
        shutter_pct = ForecastBuilder.materialize_horizon_array(
            name="shutter_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.shutter_forecast,
            fallback_scalar=forecast_config.shutter_living_room_pct,
        )

        if forecast_config.pv_enabled:
            gti_pv = ForecastBuilder.materialize_horizon_array(
                name="gti_pv_forecast",
                horizon_steps=horizon_steps,
                values=forecast_config.gti_pv_forecast,
                fallback_scalar=0.0,
            )
            pv_kw = np.maximum(gti_pv / W_PER_KW * forecast_config.pv_peak_kw, 0.0)
        else:
            pv_kw = np.zeros(horizon_steps)

        return ForecastHorizon(
            outdoor_temperature_c=t_out_arr,
            gti_w_per_m2=gti_window,
            internal_gains_kw=internal_gains,
            price_eur_per_kwh=prices,
            feed_in_price_eur_per_kwh=feed_in_prices,
            room_temperature_ref_c=np.full(horizon_steps + 1, forecast_config.room_temperature_ref_c),
            pv_kw=pv_kw,
            cop_ufh_k=cop_model.cop_ufh(t_out_arr),
            shutter_pct=shutter_pct,
        )

    @staticmethod
    def build_internal_gains_profile(req: Any, *, horizon_steps: int) -> np.ndarray:
        """Return the internal-gains horizon using explicit forecast or baseload mapping."""
        forecast_config = req.ufh_forecast_config
        if forecast_config.internal_gains_forecast is not None:
            explicit_internal_gains = ForecastBuilder.materialize_horizon_array(
                name="internal_gains_forecast",
                horizon_steps=horizon_steps,
                values=forecast_config.internal_gains_forecast,
            )
            if np.any(explicit_internal_gains < 0.0):
                raise ValueError("internal_gains_forecast must remain non-negative [kW].")
            return explicit_internal_gains

        baseline_internal_gains = np.full(horizon_steps, forecast_config.internal_gains_kw, dtype=float)
        if (
            forecast_config.baseload_forecast is None
            or forecast_config.internal_gains_heat_fraction == 0.0
        ):
            return baseline_internal_gains

        baseload_profile = ForecastBuilder.materialize_horizon_array(
            name="baseload_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.baseload_forecast,
        )
        return baseline_internal_gains + ForecastBuilder.map_baseload_to_internal_gains_increment(
            baseload_profile_kw=baseload_profile,
            baseline_internal_gains_kw=forecast_config.internal_gains_kw,
            heat_fraction=forecast_config.internal_gains_heat_fraction,
        )

    @staticmethod
    def map_baseload_to_internal_gains_increment(
        *,
        baseload_profile_kw: np.ndarray,
        baseline_internal_gains_kw: float,
        heat_fraction: float,
    ) -> np.ndarray:
        """Return the incremental sensible heat gain implied by the electrical baseload."""
        if np.any(baseload_profile_kw < 0.0):
            raise ValueError("baseload_forecast must remain non-negative [kW].")
        if heat_fraction < 0.0:
            raise ValueError("internal_gains_heat_fraction must remain non-negative [-].")
        if heat_fraction == 0.0:
            return np.zeros_like(baseload_profile_kw)
        useful_baseload_heat_kw = heat_fraction * baseload_profile_kw
        return np.maximum(useful_baseload_heat_kw - baseline_internal_gains_kw, 0.0)

    @staticmethod
    def build_dhw_forecast(
        req: "RunRequest",
        *,
        horizon_steps: int,
        cop_model: HeatPumpCOPModel,
        start_hour: int,
    ) -> DHWForecastHorizon:
        """Build the DHW disturbance and COP forecast over the horizon."""
        forecast_config = req.dhw_forecast_config
        t_out_arr = ForecastBuilder.materialize_horizon_array(
            name="t_out_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.t_out_forecast,
            fallback_scalar=forecast_config.outdoor_temperature_c,
        )
        if forecast_config.v_tap_forecast_m3_per_h is None:
            raise ValueError(
                "dhw_v_tap_forecast is required when DHW control is enabled; hidden zero-demand defaults are forbidden."
            )
        v_tap_arr = ForecastBuilder.materialize_horizon_array(
            name="dhw_v_tap_forecast",
            horizon_steps=horizon_steps,
            values=forecast_config.v_tap_forecast_m3_per_h,
        )
        target_top_arr = np.full(horizon_steps, forecast_config.t_dhw_min_c, dtype=float)
        if forecast_config.schedule_enabled:
            for step_index in range(horizon_steps):
                local_hour = (float(start_hour) + step_index * float(req.dt_hours)) % 24.0
                schedule_start = float(forecast_config.schedule_start_hour_local)
                schedule_end = schedule_start + float(forecast_config.schedule_duration_hours)
                in_schedule_window = (
                    schedule_start <= local_hour < schedule_end
                    if schedule_end <= 24.0
                    else (local_hour >= schedule_start or local_hour < (schedule_end % 24.0))
                )
                if in_schedule_window:
                    target_top_arr[step_index] = max(
                        target_top_arr[step_index],
                        forecast_config.schedule_target_c,
                    )
        elif float(req.dhw_T_top_init) < float(forecast_config.t_dhw_target_c):
            target_top_arr[:] = np.maximum(target_top_arr, forecast_config.t_dhw_target_c)
        else:
            target_top_arr[-1] = max(target_top_arr[-1], forecast_config.t_dhw_target_c)
        return DHWForecastHorizon(
            v_tap_m3_per_h=v_tap_arr,
            t_mains_c=np.full(horizon_steps, forecast_config.t_mains_c),
            t_amb_c=np.full(horizon_steps, forecast_config.t_ambient_c),
            legionella_required=np.zeros(horizon_steps, dtype=bool),
            target_top_c=target_top_arr,
            cop_dhw_k=cop_model.cop_dhw(t_out_arr, t_dhw_supply=target_top_arr),
        )


__all__ = [
    "ForecastBuilder",
    "build_repository_forecast_overrides",
    "inject_forecast_overrides",
]
