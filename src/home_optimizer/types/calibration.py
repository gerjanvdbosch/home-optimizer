"""Calibration payload models shared between the runtime and persistence layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CalibrationParameterOverrides(BaseModel):
    """Validated calibrated parameter overrides that can be applied to ``RunRequest``.

    The automatic calibration pipeline stores only parameters that are directly
    usable by the runtime MPC/COP models. Every field is optional so a stage can
    update only the parameters it actually identified while previous successful
    values remain active.
    """

    model_config = ConfigDict(extra="forbid")

    C_r: float | None = Field(default=None, gt=0.0, description="UFH room capacity C_r [kWh/K]")
    C_b: float | None = Field(default=None, gt=0.0, description="UFH slab capacity C_b [kWh/K]")
    R_br: float | None = Field(default=None, gt=0.0, description="UFH floor-room resistance R_br [K/kW]")
    R_ro: float | None = Field(default=None, gt=0.0, description="UFH room-outdoor resistance R_ro [K/kW]")
    eta: float | None = Field(default=None, ge=0.0, le=1.0, description="UFH glazing solar transmittance eta [-]")
    internal_gains_heat_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Useful indoor heat fraction of household baseload [-]",
    )
    room_temperature_bias_c: float | None = Field(
        default=None,
        description="Additive room-temperature sensor bias correction [°C]",
    )
    dhw_C_top: float | None = Field(default=None, gt=0.0, description="DHW top-layer thermal capacity C_top [kWh/K]")
    dhw_C_bot: float | None = Field(default=None, gt=0.0, description="DHW bottom-layer thermal capacity C_bot [kWh/K]")
    dhw_R_strat: float | None = Field(default=None, gt=0.0, description="DHW stratification resistance R_strat [K/kW]")
    dhw_R_loss: float | None = Field(default=None, gt=0.0, description="DHW standby-loss resistance R_loss [K/kW]")
    dhw_top_temperature_bias_c: float | None = Field(
        default=None,
        description="Additive DHW top-temperature sensor bias correction [°C]",
    )
    dhw_bottom_temperature_bias_c: float | None = Field(
        default=None,
        description="Additive DHW bottom-temperature sensor bias correction [°C]",
    )
    dhw_boiler_ambient_bias_c: float | None = Field(
        default=None,
        description="Additive DHW boiler-ambient sensor bias correction [°C]",
    )
    eta_carnot: float | None = Field(default=None, gt=0.0, le=1.0, description="Shared Carnot efficiency eta_carnot [-]")
    T_supply_min: float | None = Field(default=None, description="UFH minimum supply temperature T_supply_min [°C]")
    T_ref_outdoor_curve: float | None = Field(
        default=None,
        description="UFH heating-curve balance-point outdoor temperature T_ref_outdoor [°C]",
    )
    heating_curve_slope: float | None = Field(default=None, ge=0.0, description="UFH heating-curve slope [K/K]")

    def as_run_request_updates(self) -> dict[str, float]:
        """Return only the non-null fields as ``RunRequest.model_copy`` updates."""
        return self.model_dump(exclude_none=True)

    def merged_with(self, newer: "CalibrationParameterOverrides") -> "CalibrationParameterOverrides":
        """Return ``self`` with any non-null values from ``newer`` applied."""
        return type(self).model_validate({**self.model_dump(), **newer.model_dump(exclude_none=True)})


class CalibrationStageResult(BaseModel):
    """Summary of one automatic calibration stage."""

    model_config = ConfigDict(extra="forbid")

    stage_name: str = Field(min_length=1)
    succeeded: bool
    message: str = Field(min_length=1)
    sample_count: int | None = Field(default=None, ge=0)
    segment_count: int | None = Field(default=None, ge=0)
    dataset_start_utc: datetime | None = None
    dataset_end_utc: datetime | None = None
    optimizer_status: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    overrides: CalibrationParameterOverrides = Field(default_factory=CalibrationParameterOverrides)


class CalibrationSnapshotPayload(BaseModel):
    """Persisted automatic-calibration snapshot consumed by the scheduled MPC path."""

    model_config = ConfigDict(extra="forbid")

    generated_at_utc: datetime
    effective_parameters: CalibrationParameterOverrides = Field(default_factory=CalibrationParameterOverrides)
    ufh_active: CalibrationStageResult | None = None
    dhw_standby: CalibrationStageResult | None = None
    dhw_active: CalibrationStageResult | None = None
    cop: CalibrationStageResult | None = None

    @property
    def has_effective_parameters(self) -> bool:
        """Return ``True`` when at least one calibrated override is available."""
        return bool(self.effective_parameters.as_run_request_updates())


__all__ = [
    "CalibrationParameterOverrides",
    "CalibrationSnapshotPayload",
    "CalibrationStageResult",
]

