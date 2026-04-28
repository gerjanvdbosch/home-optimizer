from __future__ import annotations

from home_optimizer.domain.models import DomainModel


class ThermalModelCoefficients(DomainModel):
    intercept: float
    room_temperature: float
    outdoor_temperature: float
    heatpump_power: float
    solar_gain: float


class IdentificationMetrics(DomainModel):
    sample_count: int
    rmse: float
    mae: float
    r_squared: float


class ThermalModelIdentificationResult(DomainModel):
    target_name: str
    input_names: list[str]
    sample_interval_minutes: int
    coefficients: ThermalModelCoefficients
    metrics: IdentificationMetrics
