"""Architecture-focused tests for application-layer forecast and solve builders."""

from __future__ import annotations

import numpy as np

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


def test_optimizer_reexports_run_request_from_models_module() -> None:
    """Legacy imports from optimizer.py should still point at the canonical request model."""
    assert RunRequest is RunRequestModel


def test_optimizer_runtime_build_scheduled_input_preserves_base_request_without_overrides() -> None:
    """Runtime scheduled-input builder should return the validated base request when no overrides exist."""
    request = RunRequest.model_validate({"horizon_hours": 4})

    scheduled = OptimizerRuntime.build_scheduled_input(
        base_input=request,
        backend=None,
        repository=None,
    )

    assert scheduled == request
