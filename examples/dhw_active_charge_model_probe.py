"""Probe whether active-DHW telemetry warrants a richer quasi-mixed charging model.

This example is intentionally diagnostic-only: it does **not** change runtime MPC or
calibration behaviour. Instead, it compares the current active-DHW replay model
(section 9–11 implementation: all charging power enters the bottom layer) against a
strictly energy-conserving richer candidate where a fitted fraction of the charging
power is delivered directly to the top layer.

Why this candidate?
-------------------
The current active calibration assumes

    Q_charge,top = 0
    Q_charge,bot = P_dhw

which is physically correct for an ideal bottom heat exchanger with no charge-induced
internal recirculation. Real boilers can behave more like a quasi-mixed charging
process: pump circulation and internal remixing can make the top sensor warm up
faster than the pure bottom-only model predicts. A split-charge model captures that
behaviour without violating the first law:

    Q_charge,top = beta_charge * P_dhw
    Q_charge,bot = (1 - beta_charge) * P_dhw

with ``0 <= beta_charge <= 1``.

The probe answers a practical design question: does this richer model reduce replay
residuals enough on real telemetry to justify promoting it into the production
calibration path?

Units: power [kW], temperature [°C], energy [kWh], time [h].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from home_optimizer.calibration.dhw_active import calibrate_dhw_active_stratification
from home_optimizer.calibration.service import (
    _infer_calibration_replay_dt_hours,
    build_dhw_active_dataset_from_repository,
    calibrate_dhw_standby_from_repository,
)
from home_optimizer.calibration.settings_factory import (
    build_dhw_active_calibration_settings,
    build_dhw_standby_calibration_settings,
)
from home_optimizer.application.optimizer import RunRequest
from home_optimizer.telemetry.repository import TelemetryRepository
from home_optimizer.types import DHWParameters


DEFAULT_PROBE_TANK_VOLUME_M3: float = 0.2


class ProbeScenario(StrEnum):
    """Supported reference-parameter scenarios for the active-DHW probe."""

    RUNTIME_DEFAULT = "runtime-default"
    TANK_200L = "tank-200l"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    """Aggregate replay metrics for one candidate model.

    Attributes:
        rmse_t_top_c: Top-layer RMSE over the selected active-DHW replay samples [°C].
        rmse_t_bot_c: Bottom-layer RMSE over the selected active-DHW replay samples [°C].
        rmse_total_c: Combined RMSE across both layers [°C].
        mean_t_top_residual_c: Signed mean top residual ``T_top,pred - T_top,meas`` [°C].
        mean_t_bot_residual_c: Signed mean bottom residual ``T_bot,pred - T_bot,meas`` [°C].
        max_abs_residual_c: Maximum absolute residual over both layers [°C].
    """

    rmse_t_top_c: float
    rmse_t_bot_c: float
    rmse_total_c: float
    mean_t_top_residual_c: float
    mean_t_bot_residual_c: float
    max_abs_residual_c: float


@dataclass(frozen=True, slots=True)
class BaselineProbeResult:
    """Fit result for the current bottom-only active-DHW replay model."""

    fitted_parameters: DHWParameters
    metrics: ReplayMetrics


@dataclass(frozen=True, slots=True)
class SplitChargeProbeResult:
    """Fit result for the richer split-charge active-DHW replay model.

    Attributes:
        fitted_parameters: DHW parameters with fitted ``R_strat`` [K/kW].
        beta_charge_top: Fraction of charging power delivered to the top layer [-].
        metrics: Replay-quality diagnostics for the fitted richer model.
        optimizer_cost: Nonlinear least-squares cost (half SSE) [-].
        optimizer_status: SciPy termination message for traceability.
    """

    fitted_parameters: DHWParameters
    beta_charge_top: float
    metrics: ReplayMetrics
    optimizer_cost: float
    optimizer_status: str


@dataclass(frozen=True, slots=True)
class ChargeGainProbeResult:
    """Fit result for a common-mode charging-gain diagnostic model.

    The model keeps bottom-only heating, but fits a scalar multiplier on the
    measured charging power. If this probe improves the replay much more than the
    split-charge probe, the dominant mismatch is common-mode injected energy rather
    than missing top/bottom redistribution physics.
    """

    fitted_parameters: DHWParameters
    charge_gain: float
    metrics: ReplayMetrics
    optimizer_cost: float
    optimizer_status: str


@dataclass(frozen=True, slots=True)
class ScenarioProbeSummary:
    """Combined baseline-versus-richer-model summary for one telemetry scenario."""

    scenario_name: str
    sample_count: int
    segment_count: int
    reference_parameters: DHWParameters
    baseline: BaselineProbeResult
    split_charge: SplitChargeProbeResult
    charge_gain: ChargeGainProbeResult


def _exact_step_with_charge_split(
    *,
    state_c: np.ndarray,
    parameters: DHWParameters,
    control_kw: float,
    t_amb_c: float,
    t_mains_c: float,
    v_tap_m3_per_h: float,
    beta_charge_top: float,
    dt_hours: float,
) -> np.ndarray:
    """Propagate one DHW step with an explicit top/bottom charge split.

    This keeps the same continuous losses and inter-layer transfer as
    :class:`home_optimizer.domain.dhw.model.DHWModel`, but replaces assumption A5's pure
    bottom injection by the richer diagnostic split

    ``Q_charge,top = beta_charge_top * P_dhw`` and
    ``Q_charge,bot = (1 - beta_charge_top) * P_dhw``.

    The step is discretised exactly under zero-order hold, matching the runtime DHW
    model's exact-ZOH philosophy.
    """

    if state_c.shape != (2,):
        raise ValueError("state_c must be [T_top, T_bot].")
    if not 0.0 <= beta_charge_top <= 1.0:
        raise ValueError("beta_charge_top must lie in [0, 1].")
    if v_tap_m3_per_h < 0.0:
        raise ValueError("v_tap_m3_per_h must be non-negative.")
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be strictly positive.")

    strat_top_per_h = 1.0 / (parameters.C_top * parameters.R_strat)
    strat_bot_per_h = 1.0 / (parameters.C_bot * parameters.R_strat)
    loss_top_per_h = 1.0 / (parameters.C_top * parameters.R_loss)
    loss_bot_per_h = 1.0 / (parameters.C_bot * parameters.R_loss)
    tap_top_per_h = parameters.lambda_water * v_tap_m3_per_h / parameters.C_top
    tap_bot_per_h = parameters.lambda_water * v_tap_m3_per_h / parameters.C_bot

    # Implements the richer continuous charging physics used only for this probe.
    f_matrix_per_h = np.array(
        [
            [-(strat_top_per_h + loss_top_per_h + tap_top_per_h), strat_top_per_h],
            [strat_bot_per_h, -(strat_bot_per_h + loss_bot_per_h)],
        ],
        dtype=float,
    )
    g_u = np.array(
        [[beta_charge_top / parameters.C_top], [(1.0 - beta_charge_top) / parameters.C_bot]],
        dtype=float,
    )
    g_d = np.array(
        [
            [loss_top_per_h, 0.0],
            [loss_bot_per_h, tap_bot_per_h],
        ],
        dtype=float,
    )

    augmented = np.zeros((5, 5), dtype=float)
    augmented[:2, :2] = f_matrix_per_h
    augmented[:2, 2:3] = g_u
    augmented[:2, 3:] = g_d
    discretised = expm(augmented * dt_hours)

    a_matrix = discretised[:2, :2]
    b_matrix = discretised[:2, 2:3]
    e_matrix = discretised[:2, 3:]
    return a_matrix @ state_c + b_matrix[:, 0] * control_kw + e_matrix @ np.array([t_amb_c, t_mains_c], dtype=float)


def _replay_metrics_for_model(
    *,
    dataset,
    parameters: DHWParameters,
    beta_charge_top: float,
    charge_gain: float = 1.0,
) -> ReplayMetrics:
    """Return replay metrics for the diagnostic split-charge model on one dataset."""

    t_top_residuals_c: list[float] = []
    t_bot_residuals_c: list[float] = []
    for sample in dataset.samples:
        predicted_state_c = _exact_step_with_charge_split(
            state_c=np.array([sample.t_top_start_c, sample.t_bot_start_c], dtype=float),
            parameters=parameters,
            control_kw=charge_gain * sample.p_dhw_mean_kw,
            t_amb_c=sample.t_amb_c,
            t_mains_c=sample.t_mains_c,
            v_tap_m3_per_h=0.0,
            beta_charge_top=beta_charge_top,
            dt_hours=sample.dt_hours,
        )
        t_top_residuals_c.append(float(predicted_state_c[0] - sample.t_top_end_c))
        t_bot_residuals_c.append(float(predicted_state_c[1] - sample.t_bot_end_c))

    top_residuals = np.array(t_top_residuals_c, dtype=float)
    bot_residuals = np.array(t_bot_residuals_c, dtype=float)
    combined_residuals = np.concatenate([top_residuals, bot_residuals])
    return ReplayMetrics(
        rmse_t_top_c=float(np.sqrt(np.mean(np.square(top_residuals)))),
        rmse_t_bot_c=float(np.sqrt(np.mean(np.square(bot_residuals)))),
        rmse_total_c=float(np.sqrt(np.mean(np.square(combined_residuals)))),
        mean_t_top_residual_c=float(np.mean(top_residuals)),
        mean_t_bot_residual_c=float(np.mean(bot_residuals)),
        max_abs_residual_c=float(np.max(np.abs(combined_residuals))),
    )


def _fit_split_charge_model(*, dataset, reference_parameters: DHWParameters) -> SplitChargeProbeResult:
    """Fit ``(R_strat, beta_charge_top)`` on the selected active-DHW dataset.

    The fit is intentionally constrained to two physically interpretable degrees of
    freedom only:

    * ``R_strat > 0`` [K/kW]
    * ``0 <= beta_charge_top <= 1`` [-]

    This avoids hiding model mismatch inside an over-parameterised diagnostic model.
    """

    def residual_vector(theta: np.ndarray) -> np.ndarray:
        r_strat_k_per_kw = float(theta[0])
        beta_charge_top = float(theta[1])
        if r_strat_k_per_kw <= 0.0 or not 0.0 <= beta_charge_top <= 1.0:
            return np.full(2 * dataset.sample_count, 1_000.0, dtype=float)

        candidate_parameters = DHWParameters(
            dt_hours=reference_parameters.dt_hours,
            C_top=reference_parameters.C_top,
            C_bot=reference_parameters.C_bot,
            R_strat=r_strat_k_per_kw,
            R_loss=reference_parameters.R_loss,
            lambda_water=reference_parameters.lambda_water,
        )

        residuals_c: list[float] = []
        for sample in dataset.samples:
            predicted_state_c = _exact_step_with_charge_split(
                state_c=np.array([sample.t_top_start_c, sample.t_bot_start_c], dtype=float),
                parameters=candidate_parameters,
                control_kw=sample.p_dhw_mean_kw,
                t_amb_c=sample.t_amb_c,
                t_mains_c=sample.t_mains_c,
                v_tap_m3_per_h=0.0,
                beta_charge_top=beta_charge_top,
                dt_hours=sample.dt_hours,
            )
            residuals_c.extend(
                [
                    float(predicted_state_c[0] - sample.t_top_end_c),
                    float(predicted_state_c[1] - sample.t_bot_end_c),
                ]
            )
        return np.array(residuals_c, dtype=float)

    result = least_squares(
        residual_vector,
        x0=np.array([reference_parameters.R_strat, 0.1], dtype=float),
        bounds=(np.array([1e-3, 0.0], dtype=float), np.array([50.0, 1.0], dtype=float)),
        method="trf",
    )
    fitted_parameters = DHWParameters(
        dt_hours=reference_parameters.dt_hours,
        C_top=reference_parameters.C_top,
        C_bot=reference_parameters.C_bot,
        R_strat=float(result.x[0]),
        R_loss=reference_parameters.R_loss,
        lambda_water=reference_parameters.lambda_water,
    )
    beta_charge_top = float(result.x[1])
    return SplitChargeProbeResult(
        fitted_parameters=fitted_parameters,
        beta_charge_top=beta_charge_top,
        metrics=_replay_metrics_for_model(
            dataset=dataset,
            parameters=fitted_parameters,
            beta_charge_top=beta_charge_top,
        ),
        optimizer_cost=float(result.cost),
        optimizer_status=str(result.message),
    )


def _fit_charge_gain_model(*, dataset, reference_parameters: DHWParameters) -> ChargeGainProbeResult:
    """Fit ``(R_strat, charge_gain)`` for a common-mode heating-bias diagnosis."""

    def residual_vector(theta: np.ndarray) -> np.ndarray:
        r_strat_k_per_kw = float(theta[0])
        charge_gain = float(theta[1])
        if r_strat_k_per_kw <= 0.0 or charge_gain <= 0.0:
            return np.full(2 * dataset.sample_count, 1_000.0, dtype=float)

        candidate_parameters = DHWParameters(
            dt_hours=reference_parameters.dt_hours,
            C_top=reference_parameters.C_top,
            C_bot=reference_parameters.C_bot,
            R_strat=r_strat_k_per_kw,
            R_loss=reference_parameters.R_loss,
            lambda_water=reference_parameters.lambda_water,
        )

        residuals_c: list[float] = []
        for sample in dataset.samples:
            predicted_state_c = _exact_step_with_charge_split(
                state_c=np.array([sample.t_top_start_c, sample.t_bot_start_c], dtype=float),
                parameters=candidate_parameters,
                control_kw=charge_gain * sample.p_dhw_mean_kw,
                t_amb_c=sample.t_amb_c,
                t_mains_c=sample.t_mains_c,
                v_tap_m3_per_h=0.0,
                beta_charge_top=0.0,
                dt_hours=sample.dt_hours,
            )
            residuals_c.extend(
                [
                    float(predicted_state_c[0] - sample.t_top_end_c),
                    float(predicted_state_c[1] - sample.t_bot_end_c),
                ]
            )
        return np.array(residuals_c, dtype=float)

    result = least_squares(
        residual_vector,
        x0=np.array([reference_parameters.R_strat, 1.0], dtype=float),
        bounds=(np.array([1e-3, 0.1], dtype=float), np.array([50.0, 5.0], dtype=float)),
        method="trf",
    )
    fitted_parameters = DHWParameters(
        dt_hours=reference_parameters.dt_hours,
        C_top=reference_parameters.C_top,
        C_bot=reference_parameters.C_bot,
        R_strat=float(result.x[0]),
        R_loss=reference_parameters.R_loss,
        lambda_water=reference_parameters.lambda_water,
    )
    charge_gain = float(result.x[1])
    return ChargeGainProbeResult(
        fitted_parameters=fitted_parameters,
        charge_gain=charge_gain,
        metrics=_replay_metrics_for_model(
            dataset=dataset,
            parameters=fitted_parameters,
            beta_charge_top=0.0,
            charge_gain=charge_gain,
        ),
        optimizer_cost=float(result.cost),
        optimizer_status=str(result.message),
    )


def _build_reference_parameters(
    *,
    repository: TelemetryRepository,
    scenario: ProbeScenario,
    dt_hours: float,
    default_request: RunRequest,
    tank_volume_m3: float,
) -> tuple[str, DHWParameters]:
    """Return one reference parameter set for the requested probe scenario."""

    if scenario is ProbeScenario.RUNTIME_DEFAULT:
        standby_result = calibrate_dhw_standby_from_repository(
            repository,
            build_dhw_standby_calibration_settings(
                dt_hours=dt_hours,
                reference_c_top_kwh_per_k=default_request.dhw_C_top,
                reference_c_bot_kwh_per_k=default_request.dhw_C_bot,
            ),
        )
        return (
            "runtime-default",
            DHWParameters(
                dt_hours=dt_hours,
                C_top=default_request.dhw_C_top,
                C_bot=default_request.dhw_C_bot,
                R_strat=default_request.dhw_R_strat,
                R_loss=standby_result.suggested_r_loss_k_per_kw,
                lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
            ),
        )

    if scenario is ProbeScenario.TANK_200L:
        layer_capacity_kwh_per_k = tank_volume_m3 * default_request.dhw_lambda_water_kwh_per_m3k / 2.0
        standby_result = calibrate_dhw_standby_from_repository(
            repository,
            build_dhw_standby_calibration_settings(
                dt_hours=dt_hours,
                reference_c_top_kwh_per_k=layer_capacity_kwh_per_k,
                reference_c_bot_kwh_per_k=layer_capacity_kwh_per_k,
            ),
        )
        return (
            "tank-200l",
            DHWParameters(
                dt_hours=dt_hours,
                C_top=layer_capacity_kwh_per_k,
                C_bot=layer_capacity_kwh_per_k,
                R_strat=default_request.dhw_R_strat,
                R_loss=standby_result.suggested_r_loss_k_per_kw,
                lambda_water=default_request.dhw_lambda_water_kwh_per_m3k,
            ),
        )

    raise ValueError(f"Unsupported scenario: {scenario}")


def _probe_single_scenario(
    *,
    repository: TelemetryRepository,
    scenario: ProbeScenario,
    dt_hours: float,
    default_request: RunRequest,
    tank_volume_m3: float,
) -> ScenarioProbeSummary:
    """Fit and compare baseline versus richer split-charge replay for one scenario."""

    scenario_name, reference_parameters = _build_reference_parameters(
        repository=repository,
        scenario=scenario,
        dt_hours=dt_hours,
        default_request=default_request,
        tank_volume_m3=tank_volume_m3,
    )
    settings = build_dhw_active_calibration_settings(reference_parameters=reference_parameters)
    dataset = build_dhw_active_dataset_from_repository(repository, settings)

    baseline_result = calibrate_dhw_active_stratification(dataset, settings)
    baseline_parameters = baseline_result.fitted_parameters
    baseline_probe = BaselineProbeResult(
        fitted_parameters=baseline_parameters,
        metrics=_replay_metrics_for_model(
            dataset=dataset,
            parameters=baseline_parameters,
            beta_charge_top=0.0,
        ),
    )
    split_charge_probe = _fit_split_charge_model(
        dataset=dataset,
        reference_parameters=reference_parameters,
    )
    charge_gain_probe = _fit_charge_gain_model(
        dataset=dataset,
        reference_parameters=reference_parameters,
    )
    return ScenarioProbeSummary(
        scenario_name=scenario_name,
        sample_count=dataset.sample_count,
        segment_count=dataset.segment_count,
        reference_parameters=reference_parameters,
        baseline=baseline_probe,
        split_charge=split_charge_probe,
        charge_gain=charge_gain_probe,
    )


def _print_summary(summary: ScenarioProbeSummary) -> None:
    """Render one scenario summary in a compact traceable text format."""

    baseline = summary.baseline
    split_charge = summary.split_charge
    charge_gain = summary.charge_gain
    split_rmse_improvement_c = baseline.metrics.rmse_total_c - split_charge.metrics.rmse_total_c
    split_rmse_improvement_pct = (
        np.nan
        if baseline.metrics.rmse_total_c <= 0.0
        else 100.0 * split_rmse_improvement_c / baseline.metrics.rmse_total_c
    )
    gain_rmse_improvement_c = baseline.metrics.rmse_total_c - charge_gain.metrics.rmse_total_c
    gain_rmse_improvement_pct = (
        np.nan
        if baseline.metrics.rmse_total_c <= 0.0
        else 100.0 * gain_rmse_improvement_c / baseline.metrics.rmse_total_c
    )

    print(f"scenario={summary.scenario_name}")
    print(
        {
            "sample_count": summary.sample_count,
            "segment_count": summary.segment_count,
            "reference_C_top_kwh_per_k": summary.reference_parameters.C_top,
            "reference_C_bot_kwh_per_k": summary.reference_parameters.C_bot,
            "reference_R_loss_k_per_kw": summary.reference_parameters.R_loss,
        }
    )
    print(
        {
            "model": "baseline_bottom_only",
            "fitted_R_strat_k_per_kw": baseline.fitted_parameters.R_strat,
            "rmse_t_top_c": baseline.metrics.rmse_t_top_c,
            "rmse_t_bot_c": baseline.metrics.rmse_t_bot_c,
            "rmse_total_c": baseline.metrics.rmse_total_c,
            "mean_t_top_residual_c": baseline.metrics.mean_t_top_residual_c,
            "mean_t_bot_residual_c": baseline.metrics.mean_t_bot_residual_c,
            "max_abs_residual_c": baseline.metrics.max_abs_residual_c,
        }
    )
    print(
        {
            "model": "split_charge_probe",
            "fitted_R_strat_k_per_kw": split_charge.fitted_parameters.R_strat,
            "beta_charge_top": split_charge.beta_charge_top,
            "rmse_t_top_c": split_charge.metrics.rmse_t_top_c,
            "rmse_t_bot_c": split_charge.metrics.rmse_t_bot_c,
            "rmse_total_c": split_charge.metrics.rmse_total_c,
            "mean_t_top_residual_c": split_charge.metrics.mean_t_top_residual_c,
            "mean_t_bot_residual_c": split_charge.metrics.mean_t_bot_residual_c,
            "max_abs_residual_c": split_charge.metrics.max_abs_residual_c,
            "optimizer_cost": split_charge.optimizer_cost,
            "optimizer_status": split_charge.optimizer_status,
        }
    )
    print(
        {
            "model": "charge_gain_probe",
            "fitted_R_strat_k_per_kw": charge_gain.fitted_parameters.R_strat,
            "charge_gain": charge_gain.charge_gain,
            "rmse_t_top_c": charge_gain.metrics.rmse_t_top_c,
            "rmse_t_bot_c": charge_gain.metrics.rmse_t_bot_c,
            "rmse_total_c": charge_gain.metrics.rmse_total_c,
            "mean_t_top_residual_c": charge_gain.metrics.mean_t_top_residual_c,
            "mean_t_bot_residual_c": charge_gain.metrics.mean_t_bot_residual_c,
            "max_abs_residual_c": charge_gain.metrics.max_abs_residual_c,
            "optimizer_cost": charge_gain.optimizer_cost,
            "optimizer_status": charge_gain.optimizer_status,
        }
    )
    print(
        {
            "split_charge_rmse_total_improvement_c": split_rmse_improvement_c,
            "split_charge_rmse_total_improvement_pct": split_rmse_improvement_pct,
            "charge_gain_rmse_total_improvement_c": gain_rmse_improvement_c,
            "charge_gain_rmse_total_improvement_pct": gain_rmse_improvement_pct,
        }
    )
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the active-DHW charge-model probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default="sqlite:///database.sqlite3",
        help="Telemetry repository URL used for the replay probe.",
    )
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in ProbeScenario],
        default=ProbeScenario.BOTH.value,
        help="Reference-parameter scenario to probe.",
    )
    parser.add_argument(
        "--tank-volume-m3",
        type=float,
        default=DEFAULT_PROBE_TANK_VOLUME_M3,
        help="Total DHW tank volume used by the tank-200l probe scenario [m³].",
    )
    return parser.parse_args()


def main() -> None:
    """Run the active-DHW quasi-mixed charging probe on persisted telemetry."""

    args = _parse_args()
    repository = TelemetryRepository(database_url=args.database_url)
    dt_hours = _infer_calibration_replay_dt_hours(repository)
    default_request = RunRequest.model_validate({})
    requested_scenario = ProbeScenario(args.scenario)
    scenarios = (
        [ProbeScenario.RUNTIME_DEFAULT, ProbeScenario.TANK_200L]
        if requested_scenario is ProbeScenario.BOTH
        else [requested_scenario]
    )
    for scenario in scenarios:
        summary = _probe_single_scenario(
            repository=repository,
            scenario=scenario,
            dt_hours=dt_hours,
            default_request=default_request,
            tank_volume_m3=args.tank_volume_m3,
        )
        _print_summary(summary)


if __name__ == "__main__":
    main()

