"""Application-layer orchestration services for Home Optimizer."""

from .forecasting import ForecastBuilder, build_repository_forecast_overrides, inject_forecast_overrides
from .models import MPCStepResult, RunRequest, ScheduledRunSnapshot
from .optimizer import Optimizer
from .pipeline import OptimizerPipeline, OptimizerSolveContext
from .request_handling import (
    build_safe_calibration_overrides,
    merge_run_request_updates,
    sanitize_calibration_overrides,
    validate_run_request_physics,
)
from .request_projection import (
    DhwControlConfig,
    DhwForecastConfig,
    DhwPhysicalConfig,
    SharedHeatPumpConfig,
    UfhControlConfig,
    UfhForecastConfig,
    UfhPhysicalConfig,
)

__all__ = [
    "DhwControlConfig",
    "DhwForecastConfig",
    "DhwPhysicalConfig",
    "ForecastBuilder",
    "MPCStepResult",
    "Optimizer",
    "OptimizerPipeline",
    "OptimizerSolveContext",
    "RunRequest",
    "ScheduledRunSnapshot",
    "SharedHeatPumpConfig",
    "UfhControlConfig",
    "UfhForecastConfig",
    "UfhPhysicalConfig",
    "build_safe_calibration_overrides",
    "build_repository_forecast_overrides",
    "inject_forecast_overrides",
    "merge_run_request_updates",
    "sanitize_calibration_overrides",
    "validate_run_request_physics",
]
