"""Shared physical constants and numerical-validation helpers for Home Optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: 1 kW = 1000 W — used to convert W/m² irradiance to kW.
W_PER_KW: float = 1000.0

#: Absolute zero expressed in degrees Celsius. Temperatures below this are non-physical.
ABSOLUTE_ZERO_C: float = -273.15

#: Water volumetric heat capacity λ = ρ·c_p [kWh/(m³·K)] (§8.4, §15).
LAMBDA_WATER_KWH_PER_M3_K: float = 1.1628

#: Number of litres in one cubic metre [L/m³].
LITERS_PER_CUBIC_METER: float = 1000.0

#: Cubic metres in one litre [m³/L].
M3_PER_LITER: float = 1.0 / LITERS_PER_CUBIC_METER

ObservabilityConditionPolicy = Literal["min_singular_value", "max_condition_number"]


@dataclass(frozen=True, slots=True)
class NumericalValidationConfig:
    """Configurable numerical tolerances used by runtime validation and tests."""

    observability_rank_tolerance: float = 1e-9
    observability_condition_policy: ObservabilityConditionPolicy = "min_singular_value"
    observability_condition_min_sv: float = 1e-9
    observability_condition_max: float = 1e12
    covariance_psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.observability_rank_tolerance <= 0.0:
            raise ValueError("observability_rank_tolerance must be strictly positive.")
        if self.observability_condition_policy not in {"min_singular_value", "max_condition_number"}:
            raise ValueError(
                "observability_condition_policy must be 'min_singular_value' or 'max_condition_number'."
            )
        if self.observability_condition_min_sv <= 0.0:
            raise ValueError("observability_condition_min_sv must be strictly positive.")
        if self.observability_condition_max <= 1.0:
            raise ValueError("observability_condition_max must be > 1.")
        if self.covariance_psd_tolerance <= 0.0:
            raise ValueError("covariance_psd_tolerance must be strictly positive.")


DEFAULT_NUMERICAL_VALIDATION_CONFIG = NumericalValidationConfig()

__all__ = [
    "ABSOLUTE_ZERO_C",
    "DEFAULT_NUMERICAL_VALIDATION_CONFIG",
    "LAMBDA_WATER_KWH_PER_M3_K",
    "LITERS_PER_CUBIC_METER",
    "M3_PER_LITER",
    "NumericalValidationConfig",
    "ObservabilityConditionPolicy",
    "W_PER_KW",
]
