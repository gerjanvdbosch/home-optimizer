"""Validation and runtime update helpers for the application RunRequest model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..domain.dhw.model import DHWModel
from ..domain.ufh.model import ThermalModel
from ..types.calibration import CalibrationParameterOverrides

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository
    from .optimizer import RunRequest

log = logging.getLogger("home_optimizer.application.request_handling")

_UFH_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "C_r",
    "C_b",
    "R_br",
    "R_ro",
    "eta",
    "internal_gains_heat_fraction",
    "room_temperature_bias_c",
)
_DHW_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "dhw_C_top",
    "dhw_C_bot",
    "dhw_R_strat",
    "dhw_R_loss_top",
    "dhw_R_loss_bot",
    "dhw_top_temperature_bias_c",
    "dhw_bottom_temperature_bias_c",
    "dhw_boiler_ambient_bias_c",
)
_COP_CALIBRATION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "eta_carnot_ufh",
    "eta_carnot_dhw",
    "T_supply_min",
    "T_ref_outdoor_curve",
    "heating_curve_slope",
)


def validate_run_request_physics(req: "RunRequest") -> None:
    """Fail fast when a fully materialised runtime request violates coupled physics."""
    ThermalModel(req.ufh_physical_config.parameters)
    if req.dhw_physical_config.enabled:
        dhw_model = DHWModel(req.dhw_physical_config.parameters)
        dhw_model.state_matrices(0.0)


def merge_run_request_updates(base_request: "RunRequest", updates: dict[str, object]) -> "RunRequest":
    """Return a fully revalidated request after applying runtime updates."""
    if not updates:
        validate_run_request_physics(base_request)
        return base_request
    merged_request = type(base_request).model_validate(
        {**base_request.model_dump(mode="python"), **updates}
    )
    validate_run_request_physics(merged_request)
    return merged_request


def sanitize_calibration_overrides(
    base_request: "RunRequest",
    overrides: CalibrationParameterOverrides,
) -> tuple[CalibrationParameterOverrides, dict[str, str]]:
    """Keep only calibration override groups that remain valid for runtime MPC."""
    accepted_overrides = CalibrationParameterOverrides()
    rejection_reasons: dict[str, str] = {}
    override_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ufh", _UFH_CALIBRATION_OVERRIDE_FIELDS),
        ("dhw", _DHW_CALIBRATION_OVERRIDE_FIELDS),
        ("cop", _COP_CALIBRATION_OVERRIDE_FIELDS),
    )

    current_request = base_request
    raw_updates = overrides.as_run_request_updates()
    for group_name, fields in override_groups:
        group_updates = {field: raw_updates[field] for field in fields if field in raw_updates}
        if not group_updates:
            continue
        try:
            current_request = merge_run_request_updates(current_request, group_updates)
        except Exception as exc:  # noqa: BLE001
            rejection_reasons[group_name] = str(exc)
            continue
        accepted_overrides = accepted_overrides.merged_with(
            CalibrationParameterOverrides.model_validate(group_updates)
        )
    return accepted_overrides, rejection_reasons


def build_safe_calibration_overrides(
    base_request: "RunRequest",
    repository: "TelemetryRepository",
) -> dict[str, object]:
    """Return the latest runtime-safe calibration overrides for one request."""
    calibration_snapshot = repository.get_latest_calibration_snapshot()
    if calibration_snapshot is None or not calibration_snapshot.has_effective_parameters:
        return {}

    safe_overrides, rejection_reasons = sanitize_calibration_overrides(
        base_request,
        calibration_snapshot.effective_parameters,
    )
    for group_name, reason in rejection_reasons.items():
        log.warning(
            "Ignoring unsafe calibration override group '%s' for runtime MPC input: %s",
            group_name,
            reason,
        )
    return safe_overrides.as_run_request_updates()


__all__ = [
    "build_safe_calibration_overrides",
    "merge_run_request_updates",
    "sanitize_calibration_overrides",
    "validate_run_request_physics",
]
