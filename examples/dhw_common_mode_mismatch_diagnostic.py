"""Diagnose common-mode mismatch in active DHW charging telemetry.

This helper investigates three questions before we consider a richer active-DHW
charge model:

1. Do ``dhw_C_top`` and ``dhw_C_bot`` correspond to the real configured tank volume?
2. Does ``hp_thermal_power_mean_kw`` during ``dhw`` mode represent the heat that
   actually lands in the tank?
3. Is there a systematic common-mode bias between telemetry power and the tank
   energy increase plus standby losses during charging?

Why this matters
----------------
The active-DHW stratification fitter currently assumes that the measured charging
power is already the net thermal power injected into the tank. If that signal is
biased high or low, the optimisation can falsely drive ``R_strat`` towards
near-perfect mixing to compensate. This script separates:

* **capacity mismatch** — wrong ``C_top``/``C_bot`` relative to the real tank,
* **power semantics mismatch** — hydraulic output versus true tank injection,
* **tap contamination** — real draws during charging.

Units: power [kW], temperature [°C], energy [kWh], time [h], volume [L].
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
import re
from statistics import median
from typing import cast

import numpy as np

from home_optimizer.calibration.dataset import _implied_v_tap_m3_per_h
from home_optimizer.calibration.service import (
    _infer_calibration_replay_dt_hours,
    _load_calibration_aggregates,
    calibrate_dhw_standby_from_repository,
)
from home_optimizer.calibration.settings_factory import (
    build_dhw_active_calibration_settings,
    build_dhw_standby_calibration_settings,
)
from home_optimizer.optimizer import RunRequest
from home_optimizer.telemetry.models import TelemetryAggregate
from home_optimizer.telemetry.repository import TelemetryRepository
from home_optimizer.types import DHWParameters, LITERS_PER_CUBIC_METER

DEFAULT_CONFIG_PATH: str = "config.yaml"
DEFAULT_TANK_VOLUME_REGEX: str = r"^\s*boiler_tank_liters:\s*(?P<liters>[-+]?\d+(?:\.\d+)?)\s*$"


class CapacityScenario(StrEnum):
    """Supported DHW capacity assumptions for the common-mode diagnostic."""

    RUNTIME_DEFAULT = "runtime-default"
    CONFIGURED_TANK = "configured-tank"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class CapacitySummary:
    """Summary of one DHW capacity assumption.

    Attributes:
        scenario_name: Human-readable scenario identifier.
        c_top_kwh_per_k: Top-layer capacity assumption [kWh/K].
        c_bot_kwh_per_k: Bottom-layer capacity assumption [kWh/K].
        equivalent_tank_volume_liters: Equivalent total tank volume implied by
            ``(C_top + C_bot) / lambda_water`` [L].
        reference_r_loss_k_per_kw: Standby-derived loss resistance used in the
            active-DHW no-draw filters [K/kW].
    """

    scenario_name: str
    c_top_kwh_per_k: float
    c_bot_kwh_per_k: float
    equivalent_tank_volume_liters: float
    reference_r_loss_k_per_kw: float


@dataclass(frozen=True, slots=True)
class ChargePairBalance:
    """Per-pair tank energy balance during DHW charging.

    Attributes:
        dt_hours: Pair duration [h].
        hp_thermal_power_mean_kw: Persisted HP thermal power proxy [kW].
        inferred_tank_injection_kw: ``ΔE/Δt + Q_loss`` under the no-draw balance [kW].
        power_bias_kw: ``hp_thermal_power_mean_kw - inferred_tank_injection_kw`` [kW].
        charge_gain_ratio: ``inferred_tank_injection_kw / hp_thermal_power_mean_kw`` [-].
        implied_v_tap_m3_per_h: Draw inferred from the full-tank energy balance [m³/h].
        layer_spread_start_c: Start-of-pair top-bottom temperature spread [°C].
        layer_spread_end_c: End-of-pair top-bottom temperature spread [°C].
    """

    dt_hours: float
    hp_thermal_power_mean_kw: float
    inferred_tank_injection_kw: float
    power_bias_kw: float
    charge_gain_ratio: float
    implied_v_tap_m3_per_h: float
    layer_spread_start_c: float
    layer_spread_end_c: float


@dataclass(frozen=True, slots=True)
class ChargeBalanceSummary:
    """Aggregated common-mode mismatch statistics over multiple DHW charge pairs."""

    label: str
    pair_count: int
    mean_hp_thermal_power_kw: float
    mean_inferred_tank_injection_kw: float
    mean_power_bias_kw: float
    median_power_bias_kw: float
    mean_abs_power_bias_kw: float
    weighted_charge_gain_ratio: float
    mean_charge_gain_ratio: float
    median_charge_gain_ratio: float
    mean_implied_v_tap_m3_per_h: float
    p95_implied_v_tap_m3_per_h: float


@dataclass(frozen=True, slots=True)
class ScenarioDiagnostic:
    """Full common-mode mismatch diagnostic for one DHW capacity scenario."""

    capacity: CapacitySummary
    raw_charge_pairs: ChargeBalanceSummary
    no_draw_charge_pairs: ChargeBalanceSummary


@dataclass(frozen=True, slots=True)
class StoredThermalPowerAudit:
    """Audit summary for stored ``hp_thermal_power_*`` telemetry fields.

    Attributes:
        mode_name: Heat-pump mode covered by the audit (e.g. ``"dhw"``).
        bucket_count: Number of persisted buckets included [-].
        max_abs_last_formula_error_kw: Maximum absolute deviation between
            ``hp_thermal_power_last_kw`` and the documented hydraulic formula [kW].
        mean_last_formula_error_kw: Mean of the same last-sample deviation [kW].
        max_abs_mean_from_means_gap_kw: Maximum gap between stored
            ``hp_thermal_power_mean_kw`` and the formula applied to the bucket mean
            ``(flow_mean, supply_mean, return_mean)`` [kW]. This is expected to be
            non-zero because the persisted mean is the mean of per-sample powers, not
            the power reconstructed from aggregate means.
        mean_mean_from_means_gap_kw: Mean of that aggregate-means gap [kW].
        mean_counter_cop: Mean counter-based COP proxy
            ``hp_thermal_power_mean_kw * dt / hp_electric_energy_delta_kwh`` [-].
        mean_instantaneous_power_cop: Mean bucket-level power ratio
            ``hp_thermal_power_mean_kw / hp_electric_power_mean_kw`` [-].
    """

    mode_name: str
    bucket_count: int
    max_abs_last_formula_error_kw: float
    mean_last_formula_error_kw: float
    max_abs_mean_from_means_gap_kw: float
    mean_mean_from_means_gap_kw: float
    mean_counter_cop: float | None
    mean_instantaneous_power_cop: float | None


LITERS_PER_MINUTE_TO_M3_PER_HOUR: float = 60.0 / LITERS_PER_CUBIC_METER


def _extract_configured_tank_volume_liters(config_path: Path) -> float:
    """Read ``boiler_tank_liters`` from the addon config manifest.

    The addon manifest is intentionally simple YAML. For this diagnostic we only
    need one scalar option, so a targeted regex keeps the helper dependency-free.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    pattern = re.compile(DEFAULT_TANK_VOLUME_REGEX, flags=re.MULTILINE)
    match = pattern.search(config_path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(
            f"Could not find boiler_tank_liters in {config_path}."
        )
    tank_volume_liters = float(match.group("liters"))
    if tank_volume_liters <= 0.0:
        raise ValueError("Configured boiler_tank_liters must be strictly positive.")
    return tank_volume_liters


def _equivalent_tank_volume_liters(*, c_top_kwh_per_k: float, c_bot_kwh_per_k: float, lambda_water: float) -> float:
    """Convert total DHW heat capacity into equivalent tank volume [L]."""

    return (c_top_kwh_per_k + c_bot_kwh_per_k) / lambda_water * LITERS_PER_CUBIC_METER


def _stored_hydraulic_power_kw(*, flow_lpm: float, supply_c: float, return_c: float, lambda_water: float) -> float:
    """Return the documented hydraulic thermal power ``Vdot * lambda * dT`` [kW]."""

    return flow_lpm * LITERS_PER_MINUTE_TO_M3_PER_HOUR * lambda_water * (supply_c - return_c)


def _audit_stored_thermal_power(
    *,
    rows: list[TelemetryAggregate],
    mode_name: str,
    lambda_water: float,
) -> StoredThermalPowerAudit:
    """Audit whether persisted ``hp_thermal_power_*`` matches the documented formula."""

    mode_rows = [row for row in rows if row.hp_mode_last == mode_name]
    if not mode_rows:
        raise ValueError(f"No telemetry buckets available for mode {mode_name!r}.")

    last_formula_errors_kw: list[float] = []
    mean_from_means_gaps_kw: list[float] = []
    counter_cops: list[float] = []
    instantaneous_power_cops: list[float] = []

    for row in mode_rows:
        reconstructed_last_kw = _stored_hydraulic_power_kw(
            flow_lpm=float(row.hp_flow_last_lpm),
            supply_c=float(row.hp_supply_temperature_last_c),
            return_c=float(row.hp_return_temperature_last_c),
            lambda_water=lambda_water,
        )
        last_formula_errors_kw.append(float(row.hp_thermal_power_last_kw) - reconstructed_last_kw)

        reconstructed_mean_from_means_kw = _stored_hydraulic_power_kw(
            flow_lpm=float(row.hp_flow_mean_lpm),
            supply_c=float(row.hp_supply_temperature_mean_c),
            return_c=float(row.hp_return_temperature_mean_c),
            lambda_water=lambda_water,
        )
        mean_from_means_gaps_kw.append(float(row.hp_thermal_power_mean_kw) - reconstructed_mean_from_means_kw)

        if float(row.hp_electric_power_mean_kw) > 0.0:
            instantaneous_power_cops.append(float(row.hp_thermal_power_mean_kw) / float(row.hp_electric_power_mean_kw))

        if float(row.hp_electric_energy_delta_kwh) > 0.0:
            dt_hours = (row.bucket_end_utc - row.bucket_start_utc).total_seconds() / 3600.0
            if dt_hours > 0.0:
                counter_cops.append(float(row.hp_thermal_power_mean_kw) * dt_hours / float(row.hp_electric_energy_delta_kwh))

    return StoredThermalPowerAudit(
        mode_name=mode_name,
        bucket_count=len(mode_rows),
        max_abs_last_formula_error_kw=float(np.max(np.abs(np.array(last_formula_errors_kw, dtype=float)))),
        mean_last_formula_error_kw=float(np.mean(np.array(last_formula_errors_kw, dtype=float))),
        max_abs_mean_from_means_gap_kw=float(np.max(np.abs(np.array(mean_from_means_gaps_kw, dtype=float)))),
        mean_mean_from_means_gap_kw=float(np.mean(np.array(mean_from_means_gaps_kw, dtype=float))),
        mean_counter_cop=(float(np.mean(np.array(counter_cops, dtype=float))) if counter_cops else None),
        mean_instantaneous_power_cop=(
            float(np.mean(np.array(instantaneous_power_cops, dtype=float))) if instantaneous_power_cops else None
        ),
    )


def _build_reference_parameters(
    *,
    repository: TelemetryRepository,
    dt_hours: float,
    default_request: RunRequest,
    scenario: CapacityScenario,
    configured_tank_volume_liters: float,
) -> CapacitySummary:
    """Return one scenario-specific DHW capacity summary with standby-derived ``R_loss``."""

    if scenario is CapacityScenario.RUNTIME_DEFAULT:
        c_top_kwh_per_k = default_request.dhw_C_top
        c_bot_kwh_per_k = default_request.dhw_C_bot
        scenario_name = "runtime-default"
    elif scenario is CapacityScenario.CONFIGURED_TANK:
        total_tank_volume_m3 = configured_tank_volume_liters / LITERS_PER_CUBIC_METER
        c_top_kwh_per_k = total_tank_volume_m3 * default_request.dhw_lambda_water_kwh_per_m3k / 2.0
        c_bot_kwh_per_k = total_tank_volume_m3 * default_request.dhw_lambda_water_kwh_per_m3k / 2.0
        scenario_name = "configured-tank"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    standby_result = calibrate_dhw_standby_from_repository(
        repository,
        build_dhw_standby_calibration_settings(
            dt_hours=dt_hours,
            reference_c_top_kwh_per_k=c_top_kwh_per_k,
            reference_c_bot_kwh_per_k=c_bot_kwh_per_k,
        ),
    )
    return CapacitySummary(
        scenario_name=scenario_name,
        c_top_kwh_per_k=c_top_kwh_per_k,
        c_bot_kwh_per_k=c_bot_kwh_per_k,
        equivalent_tank_volume_liters=_equivalent_tank_volume_liters(
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
        ),
        reference_r_loss_k_per_kw=standby_result.suggested_r_loss_k_per_kw,
    )


def _tank_energy_kwh(*, t_top_c: float, t_bot_c: float, c_top_kwh_per_k: float, c_bot_kwh_per_k: float) -> float:
    """Return total tank energy proxy ``C_top*T_top + C_bot*T_bot`` [kWh]."""

    return c_top_kwh_per_k * t_top_c + c_bot_kwh_per_k * t_bot_c


def _iter_charge_pairs(
    *,
    rows: list[TelemetryAggregate],
    default_request: RunRequest,
    reference_parameters: DHWParameters,
) -> list[ChargePairBalance]:
    """Build consecutive DHW charging-pair balances under one capacity assumption."""

    settings = build_dhw_active_calibration_settings(reference_parameters=reference_parameters)
    charge_pairs: list[ChargePairBalance] = []
    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last != settings.active_mode_name:
            continue
        if next_row.hp_mode_last != settings.active_mode_name:
            continue
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / 3600.0
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            continue

        hp_thermal_power_mean_kw = float(next_row.hp_thermal_power_mean_kw)
        if hp_thermal_power_mean_kw < settings.min_dhw_power_kw:
            continue

        t_top_start_c = float(previous_row.dhw_top_temperature_last_c)
        t_top_end_c = float(next_row.dhw_top_temperature_last_c)
        t_bot_start_c = float(previous_row.dhw_bottom_temperature_last_c)
        t_bot_end_c = float(next_row.dhw_bottom_temperature_last_c)
        t_mains_c = float(next_row.t_mains_estimated_mean_c)
        t_amb_c = float(next_row.boiler_ambient_temp_mean_c)

        start_energy_kwh = _tank_energy_kwh(
            t_top_c=t_top_start_c,
            t_bot_c=t_bot_start_c,
            c_top_kwh_per_k=reference_parameters.C_top,
            c_bot_kwh_per_k=reference_parameters.C_bot,
        )
        end_energy_kwh = _tank_energy_kwh(
            t_top_c=t_top_end_c,
            t_bot_c=t_bot_end_c,
            c_top_kwh_per_k=reference_parameters.C_top,
            c_bot_kwh_per_k=reference_parameters.C_bot,
        )
        delta_energy_rate_kw = (end_energy_kwh - start_energy_kwh) / dt_hours
        mean_t_top_c = 0.5 * (t_top_start_c + t_top_end_c)
        mean_t_bot_c = 0.5 * (t_bot_start_c + t_bot_end_c)
        q_loss_kw = (
            (mean_t_top_c - t_amb_c) / reference_parameters.R_loss
            + (mean_t_bot_c - t_amb_c) / reference_parameters.R_loss
        )
        inferred_tank_injection_kw = delta_energy_rate_kw + q_loss_kw
        power_bias_kw = hp_thermal_power_mean_kw - inferred_tank_injection_kw
        charge_gain_ratio = inferred_tank_injection_kw / hp_thermal_power_mean_kw
        implied_v_tap_m3_per_h = _implied_v_tap_m3_per_h(
            t_top_start_c=t_top_start_c,
            t_bot_start_c=t_bot_start_c,
            t_top_end_c=t_top_end_c,
            t_bot_end_c=t_bot_end_c,
            dt_hours=dt_hours,
            p_dhw_mean_kw=hp_thermal_power_mean_kw,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
            settings=settings,
        )

        charge_pairs.append(
            ChargePairBalance(
                dt_hours=dt_hours,
                hp_thermal_power_mean_kw=hp_thermal_power_mean_kw,
                inferred_tank_injection_kw=inferred_tank_injection_kw,
                power_bias_kw=power_bias_kw,
                charge_gain_ratio=charge_gain_ratio,
                implied_v_tap_m3_per_h=implied_v_tap_m3_per_h,
                layer_spread_start_c=abs(t_top_start_c - t_bot_start_c),
                layer_spread_end_c=abs(t_top_end_c - t_bot_end_c),
            )
        )
    return charge_pairs


def _summarize_pairs(*, label: str, pairs: list[ChargePairBalance]) -> ChargeBalanceSummary:
    """Aggregate charge-pair balances into bias and gain statistics."""

    if not pairs:
        raise ValueError(f"No DHW charge pairs available for summary {label!r}.")

    hp_power_kw = np.array([pair.hp_thermal_power_mean_kw for pair in pairs], dtype=float)
    inferred_injection_kw = np.array([pair.inferred_tank_injection_kw for pair in pairs], dtype=float)
    power_bias_kw = np.array([pair.power_bias_kw for pair in pairs], dtype=float)
    charge_gain_ratio = np.array([pair.charge_gain_ratio for pair in pairs], dtype=float)
    implied_v_tap = np.array([pair.implied_v_tap_m3_per_h for pair in pairs], dtype=float)
    return ChargeBalanceSummary(
        label=label,
        pair_count=len(pairs),
        mean_hp_thermal_power_kw=float(np.mean(hp_power_kw)),
        mean_inferred_tank_injection_kw=float(np.mean(inferred_injection_kw)),
        mean_power_bias_kw=float(np.mean(power_bias_kw)),
        median_power_bias_kw=float(median(power_bias_kw.tolist())),
        mean_abs_power_bias_kw=float(np.mean(np.abs(power_bias_kw))),
        weighted_charge_gain_ratio=float(float(np.sum(inferred_injection_kw)) / float(np.sum(hp_power_kw))),
        mean_charge_gain_ratio=float(np.mean(charge_gain_ratio)),
        median_charge_gain_ratio=float(median(charge_gain_ratio.tolist())),
        mean_implied_v_tap_m3_per_h=float(np.mean(implied_v_tap)),
        p95_implied_v_tap_m3_per_h=float(np.percentile(implied_v_tap, 95)),
    )


def _diagnose_scenario(
    *,
    rows: list[TelemetryAggregate],
    repository: TelemetryRepository,
    dt_hours: float,
    default_request: RunRequest,
    scenario: CapacityScenario,
    configured_tank_volume_liters: float,
) -> ScenarioDiagnostic:
    """Run the full common-mode mismatch diagnosis for one capacity scenario."""

    capacity = _build_reference_parameters(
        repository=repository,
        dt_hours=dt_hours,
        default_request=default_request,
        scenario=scenario,
        configured_tank_volume_liters=configured_tank_volume_liters,
    )
    reference_parameters = DHWParameters(
        dt_hours=dt_hours,
        C_top=capacity.c_top_kwh_per_k,
        C_bot=capacity.c_bot_kwh_per_k,
        R_strat=default_request.dhw_R_strat,
        R_loss=capacity.reference_r_loss_k_per_kw,
        lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
    )
    settings = build_dhw_active_calibration_settings(reference_parameters=reference_parameters)
    raw_pairs = _iter_charge_pairs(
        rows=rows,
        default_request=default_request,
        reference_parameters=reference_parameters,
    )
    no_draw_pairs = [pair for pair in raw_pairs if pair.implied_v_tap_m3_per_h <= settings.max_implied_tap_m3_per_h]
    return ScenarioDiagnostic(
        capacity=capacity,
        raw_charge_pairs=_summarize_pairs(label="raw_dhw_charge_pairs", pairs=raw_pairs),
        no_draw_charge_pairs=_summarize_pairs(label="no_draw_charge_pairs", pairs=no_draw_pairs),
    )


def _print_capacity_check(*, default_request: RunRequest, configured_tank_volume_liters: float) -> None:
    """Print the requested capacity-versus-volume reconciliation."""

    runtime_equivalent_liters = _equivalent_tank_volume_liters(
        c_top_kwh_per_k=default_request.dhw_C_top,
        c_bot_kwh_per_k=default_request.dhw_C_bot,
        lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
    )
    configured_total_capacity_kwh_per_k = (
        configured_tank_volume_liters / LITERS_PER_CUBIC_METER * default_request.dhw_lambda_water_kwh_per_m3k
    )
    configured_layer_capacity_kwh_per_k = configured_total_capacity_kwh_per_k / 2.0
    print("capacity_check")
    print(
        {
            "runtime_default_dhw_C_top_kwh_per_k": default_request.dhw_C_top,
            "runtime_default_dhw_C_bot_kwh_per_k": default_request.dhw_C_bot,
            "runtime_default_equivalent_tank_volume_liters": runtime_equivalent_liters,
            "configured_tank_volume_liters": configured_tank_volume_liters,
            "configured_expected_dhw_C_top_kwh_per_k": configured_layer_capacity_kwh_per_k,
            "configured_expected_dhw_C_bot_kwh_per_k": configured_layer_capacity_kwh_per_k,
            "runtime_to_configured_capacity_ratio": runtime_equivalent_liters / configured_tank_volume_liters,
        }
    )
    print()


def _print_stored_thermal_power_audit(audit: StoredThermalPowerAudit) -> None:
    """Render the storage-time thermal-power audit."""

    print(f"stored_thermal_power_audit mode={audit.mode_name}")
    print(asdict(audit))
    print()


def _print_scenario_diagnostic(diagnostic: ScenarioDiagnostic) -> None:
    """Render one scenario diagnostic in a compact traceable format."""

    print(f"scenario={diagnostic.capacity.scenario_name}")
    print(
        {
            "c_top_kwh_per_k": diagnostic.capacity.c_top_kwh_per_k,
            "c_bot_kwh_per_k": diagnostic.capacity.c_bot_kwh_per_k,
            "equivalent_tank_volume_liters": diagnostic.capacity.equivalent_tank_volume_liters,
            "reference_r_loss_k_per_kw": diagnostic.capacity.reference_r_loss_k_per_kw,
        }
    )
    print(asdict(diagnostic.raw_charge_pairs))
    print(asdict(diagnostic.no_draw_charge_pairs))
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the common-mode mismatch diagnostic."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default="sqlite:///database.sqlite3",
        help="Telemetry repository URL used for the common-mode mismatch diagnostic.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Addon config manifest used to read boiler_tank_liters.",
    )
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in CapacityScenario],
        default=CapacityScenario.BOTH.value,
        help="Capacity assumption to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested common-mode mismatch checks on persisted DHW telemetry."""

    args = _parse_args()
    repository = TelemetryRepository(database_url=args.database_url)
    rows = cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository)))
    full_aggregate_rows = repository.list_aggregates()
    dt_hours = _infer_calibration_replay_dt_hours(repository)
    default_request = RunRequest.model_validate({})
    configured_tank_volume_liters = _extract_configured_tank_volume_liters(Path(args.config_path))

    _print_stored_thermal_power_audit(
        _audit_stored_thermal_power(
            rows=full_aggregate_rows,
            mode_name="dhw",
            lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
        )
    )
    _print_stored_thermal_power_audit(
        _audit_stored_thermal_power(
            rows=full_aggregate_rows,
            mode_name="ufh",
            lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
        )
    )

    _print_capacity_check(
        default_request=default_request,
        configured_tank_volume_liters=configured_tank_volume_liters,
    )

    requested_scenario = CapacityScenario(args.scenario)
    scenarios = (
        [CapacityScenario.RUNTIME_DEFAULT, CapacityScenario.CONFIGURED_TANK]
        if requested_scenario is CapacityScenario.BOTH
        else [requested_scenario]
    )
    for scenario in scenarios:
        diagnostic = _diagnose_scenario(
            rows=rows,
            repository=repository,
            dt_hours=dt_hours,
            default_request=default_request,
            scenario=scenario,
            configured_tank_volume_liters=configured_tank_volume_liters,
        )
        _print_scenario_diagnostic(diagnostic)


if __name__ == "__main__":
    main()

