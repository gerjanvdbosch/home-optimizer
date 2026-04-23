"""Architecture-focused tests for application-layer forecast and solve builders."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from home_optimizer.application.forecasting import ForecastBuilder
from home_optimizer.application.models import RunRequest as RunRequestModel
from home_optimizer.application.optimizer import RunRequest
from home_optimizer.application.pipeline import OptimizerPipeline
from home_optimizer.application.request_projection import (
    DhwForecastConfig,
    SharedHeatPumpConfig,
    UfhControlConfig,
    UfhPhysicalConfig,
)
from home_optimizer.application.runtime import OptimizerRuntime
from home_optimizer.domain.heat_pump.cop import HeatPumpCOPParameters
from home_optimizer.sensors import LiveReadings, SensorBackend


def _live_readings() -> LiveReadings:
    """Return one live telemetry snapshot for runtime-seeding tests."""
    return LiveReadings(
        room_temperature_c=20.25,
        outdoor_temperature_c=6.75,
        hp_supply_temperature_c=31.5,
        hp_supply_target_temperature_c=33.0,
        hp_return_temperature_c=27.0,
        hp_flow_lpm=8.5,
        hp_electric_power_kw=1.9,
        hp_mode="ufh",
        p1_net_power_kw=1.2,
        pv_output_kw=0.4,
        thermostat_setpoint_c=20.5,
        dhw_top_temperature_c=52.0,
        dhw_bottom_temperature_c=45.5,
        shutter_living_room_pct=78.0,
        defrost_active=False,
        booster_heater_active=False,
        boiler_ambient_temp_c=18.4,
        refrigerant_condensation_temp_c=37.8,
        refrigerant_liquid_line_temp_c=27.9,
        discharge_temp_c=64.0,
        t_mains_estimated_c=10.2,
        timestamp=datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc),
        pv_total_kwh=1234.5,
        hp_electric_total_kwh=678.9,
        p1_import_total_kwh=910.11,
        p1_export_total_kwh=12.13,
    )


class _StaticBackend(SensorBackend):
    """Deterministic backend that always returns one fixed live snapshot."""

    def __init__(self, readings: LiveReadings) -> None:
        self._readings = readings

    def read_all(self) -> LiveReadings:
        return self._readings

    def close(self) -> None:
        """Release backend resources.

        The deterministic backend has no external resources.
        """


def test_forecast_builder_materializes_scalar_fallback_to_full_horizon() -> None:
    """ForecastBuilder must broadcast scalar fallbacks without silent truncation."""
    values = ForecastBuilder.materialize_horizon_array(
        name="t_out_forecast",
        horizon_steps=4,
        values=None,
        fallback_scalar=8.0,
    )
    np.testing.assert_allclose(values, np.full(4, 8.0))


def test_optimizer_pipeline_builds_combined_context_when_dhw_enabled() -> None:
    """Pipeline context must include DHW model, forecast, and combined MPC parameters."""
    request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "dhw_enabled": True,
            "dhw_v_tap_forecast": [0.0, 0.0, 0.0, 0.0],
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
        }
    )

    context = OptimizerPipeline.build_solve_context(request, start_hour=0)

    assert context.dhw_model is not None
    assert context.dhw_forecast is not None
    assert context.initial_dhw_state_c is not None
    assert context.ufh_forecast.cop_ufh_k is not None
    assert context.dhw_forecast.cop_dhw_k is not None


def test_optimizer_pipeline_builds_cop_model_from_request_parameters() -> None:
    """Pipeline COP construction must preserve the calibrated Carnot parameter tuple."""
    request = RunRequest.model_validate(
        {
            "eta_carnot_ufh": 0.41,
            "eta_carnot_dhw": 0.38,
            "delta_T_cond": 4.0,
            "delta_T_evap": 6.0,
            "T_supply_min": 29.0,
            "T_ref_outdoor_curve": 17.0,
            "heating_curve_slope": 0.9,
            "cop_min": 1.6,
            "cop_max": 6.8,
        }
    )

    model = OptimizerPipeline.build_cop_model(request)

    assert model.params == HeatPumpCOPParameters(
        eta_carnot_ufh=0.41,
        eta_carnot_dhw=0.38,
        delta_T_cond=4.0,
        delta_T_evap=6.0,
        T_supply_min=29.0,
        T_ref_outdoor=17.0,
        heating_curve_slope=0.9,
        cop_min=1.6,
        cop_max=6.8,
    )


def test_run_request_exposes_domain_specific_projections() -> None:
    """RunRequest should provide explicit domain projections for downstream services."""
    request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "T_r_init": 20.0,
            "T_b_init": 22.0,
            "dhw_enabled": True,
            "dhw_v_tap_forecast": [0.0, 0.01, 0.0, 0.0],
        }
    )

    assert isinstance(request.ufh_physical_config, UfhPhysicalConfig)
    assert isinstance(request.ufh_control_config, UfhControlConfig)
    assert isinstance(request.dhw_forecast_config, DhwForecastConfig)
    assert isinstance(request.shared_heat_pump_config, SharedHeatPumpConfig)
    np.testing.assert_allclose(request.ufh_physical_config.initial_state_c, [20.0, 22.0])
    np.testing.assert_allclose(
        request.dhw_forecast_config.v_tap_forecast_m3_per_h,
        [0.0, 0.01, 0.0, 0.0],
    )


def test_run_request_default_dhw_capacities_match_a_200l_tank_split_over_two_nodes() -> None:
    """The bare runtime defaults should not silently assume a tank that is 5× too large."""
    request = RunRequest.model_validate({})

    np.testing.assert_allclose(
        [request.dhw_C_top, request.dhw_C_bot],
        [0.11628, 0.11628],
    )


def test_optimizer_reexports_run_request_from_models_module() -> None:
    """Legacy imports from optimizer.py should still point at the canonical request model."""
    assert RunRequest is RunRequestModel


def test_dhw_schedule_target_respects_dt_hours_across_fractional_steps() -> None:
    """The DHW target schedule must use the physical step size instead of assuming 1 h steps."""
    request = RunRequest.model_validate(
        {
            "horizon_hours": 8,
            "dt_hours": 0.25,
            "dhw_enabled": True,
            "dhw_T_min": 40.0,
            "dhw_schedule_enabled": True,
            "dhw_schedule_start_hour_local": 22,
            "dhw_schedule_duration_hours": 1,
            "dhw_schedule_target_c": 55.0,
            "dhw_v_tap_forecast": [0.0] * 8,
            "t_out_forecast": [8.0] * 8,
            "gti_window_forecast": [0.0] * 8,
            "gti_pv_forecast": [0.0] * 8,
        }
    )

    forecast = ForecastBuilder.build_dhw_forecast(
        request,
        horizon_steps=8,
        cop_model=OptimizerPipeline.build_cop_model(request),
        start_hour=21,
    )

    np.testing.assert_allclose(
        forecast.target_top_c,
        np.array([40.0, 40.0, 40.0, 40.0, 55.0, 55.0, 55.0, 55.0]),
    )


def test_optimizer_runtime_build_scheduled_input_preserves_base_request_without_overrides() -> None:
    """Runtime scheduled-input builder should return the validated base request when no overrides exist."""
    request = RunRequest.model_validate({"horizon_hours": 4})

    scheduled = OptimizerRuntime.build_scheduled_input(
        base_input=request,
        backend=None,
        repository=None,
    )

    assert scheduled == request


def test_optimizer_runtime_build_scheduled_input_seeds_live_sensors_without_tap_forecast() -> None:
    """Runtime seeding must use live sensor values but leave DHW tap demand to the repository forecast."""
    request = RunRequest.model_validate({"horizon_hours": 4, "dhw_enabled": True})

    scheduled = OptimizerRuntime.build_scheduled_input(
        base_input=request,
        backend=_StaticBackend(_live_readings()),
        repository=None,
    )

    assert scheduled.T_r_init == pytest.approx(20.25)
    assert scheduled.T_b_init == pytest.approx(29.25)
    assert scheduled.outdoor_temperature_c == pytest.approx(6.75)
    assert scheduled.shutter_living_room_pct == pytest.approx(78.0)
    assert scheduled.dhw_T_top_init == pytest.approx(52.0)
    assert scheduled.dhw_T_bot_init == pytest.approx(45.5)
    assert scheduled.dhw_t_mains_c == pytest.approx(10.2)
    assert scheduled.dhw_t_amb_c == pytest.approx(18.4)
    assert scheduled.dhw_v_tap_forecast is None


def test_run_request_accepts_exclusive_topology_without_fixed_mode_when_on_off_controls_are_enabled() -> None:
    """The simulator/addon may leave the exclusive mode empty so the mixed-integer MPC can choose it."""
    common_data = {
        "horizon_hours": 4,
        "dhw_enabled": True,
        "dhw_v_tap_forecast": [0.0] * 4,
        "t_out_forecast": [8.0] * 4,
        "gti_window_forecast": [0.0] * 4,
        "gti_pv_forecast": [0.0] * 4,
        "heat_pump_topology": "exclusive",
        "ufh_on_off_control_enabled": True,
        "dhw_on_off_control_enabled": True,
    }

    request_empty = RunRequest.model_validate(
        {
            **common_data,
            "exclusive_heat_pump_mode": "",
        }
    )
    request_null = RunRequest.model_validate(
        {
            **common_data,
            "exclusive_heat_pump_mode": None,
        }
    )

    assert request_empty.shared_heat_pump_config.topology == "exclusive"
    assert request_empty.shared_heat_pump_config.exclusive_active_mode is None
    assert request_null.shared_heat_pump_config.topology == "exclusive"
    assert request_null.shared_heat_pump_config.exclusive_active_mode is None
