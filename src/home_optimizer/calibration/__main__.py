"""Command-line entry point for offline thermal calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import (
    COP_LEAST_SQUARES_LOSS_CHOICES,
    COPCalibrationSettings,
    DEFAULT_ACTIVE_MAX_GTI_W_PER_M2,
    DEFAULT_COP_MAX,
    DEFAULT_COP_MIN,
    DEFAULT_COP_ETA_LOSS_NAME,
    DEFAULT_COP_ETA_LOSS_SCALE_KWH,
    DEFAULT_COP_HEATING_CURVE_LOSS_NAME,
    DEFAULT_COP_HEATING_CURVE_LOSS_SCALE_C,
    DEFAULT_DELTA_T_COND_K,
    DEFAULT_DELTA_T_EVAP_K,
    DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H,
    DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C,
    DEFAULT_MAX_COP_SEGMENT_SUPPLY_TRACKING_RMSE_C,
    DEFAULT_MIN_ELECTRIC_ENERGY_KWH,
    DEFAULT_MIN_COP_SEGMENT_ACTUAL_COP_SPAN,
    DEFAULT_MIN_COP_SEGMENT_SCORE,
    DEFAULT_MIN_COP_SEGMENT_SAMPLES,
    DEFAULT_MIN_COP_SEGMENT_THERMAL_ENERGY_KWH,
    DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C,
    DEFAULT_MIN_DHW_POWER_KW,
    DEFAULT_MIN_DHW_SEGMENT_SAMPLES,
    DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C,
    DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH,
    DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C,
    DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C,
    DEFAULT_MIN_DHW_SEGMENT_SCORE,
    DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C,
    DEFAULT_MIN_THERMAL_ENERGY_KWH,
    DEFAULT_MIN_UFH_CURVE_SAMPLE_COUNT,
    DEFAULT_MIN_UFH_COP_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
    DEFAULT_MIN_UFH_COP_SEGMENT_SUPPLY_TARGET_SPAN_C,
    DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C,
    DEFAULT_MIN_UFH_POWER_KW,
    DEFAULT_MIN_SEGMENT_SAMPLES,
    DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
    DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C,
    DEFAULT_MIN_SEGMENT_SCORE,
    DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW,
    DEFAULT_T_REF_OUTDOOR_C,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationSettings,
    UFHActiveCalibrationSettings,
    UFHOffCalibrationSettings,
)
from .service import (
    build_cop_dataset_from_repository,
    build_dhw_active_dataset_from_repository,
    build_dhw_standby_dataset_from_repository,
    build_ufh_active_dataset_from_repository,
    build_ufh_off_dataset_from_repository,
    calibrate_cop_from_repository,
    calibrate_dhw_active_from_repository,
    calibrate_dhw_standby_from_repository,
    calibrate_ufh_active_from_repository,
    calibrate_ufh_off_from_repository,
)
from ..telemetry.repository import TelemetryRepository
from ..types import DHWParameters, ThermalParameters

DEFAULT_DATABASE_URL: str = "sqlite:///database.sqlite3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="home-optimizer-calibration",
        description="Fit first-stage thermal parameters from persisted telemetry history.",
    )
    parser.add_argument(
        "--stage",
        choices=("off", "active-ufh", "dhw-standby", "active-dhw", "cop"),
        default="off",
        help="Calibration stage to run: passive UFH off-envelope fit, active UFH RC fit, passive DHW standby-loss fit, active DHW stratification fit, or offline COP fit.",
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="SQLAlchemy database URL containing telemetry history.",
    )
    parser.add_argument("--dt-hours", type=float, default=None, help="Reference UFH model time step Δt [h].")
    parser.add_argument("--c-r", type=float, default=None, help="Reference room capacity C_r [kWh/K].")
    parser.add_argument("--c-b", type=float, default=None, help="Reference floor capacity C_b [kWh/K].")
    parser.add_argument("--r-br", type=float, default=None, help="Reference floor-room resistance R_br [K/kW].")
    parser.add_argument("--r-ro", type=float, default=None, help="Reference room-outdoor resistance R_ro [K/kW].")
    parser.add_argument("--alpha", type=float, default=None, help="Reference solar split α [-].")
    parser.add_argument("--eta", type=float, default=None, help="Reference glazing transmittance η [-].")
    parser.add_argument("--a-glass", type=float, default=None, help="Reference glazing area A_glass [m²].")
    parser.add_argument(
        "--dhw-dt-hours",
        type=float,
        default=None,
        help="Reference DHW standby-calibration time step Δt [h].",
    )
    parser.add_argument(
        "--dhw-c-top",
        type=float,
        default=None,
        help="Reference DHW top-layer capacity C_top [kWh/K].",
    )
    parser.add_argument(
        "--dhw-c-bot",
        type=float,
        default=None,
        help="Reference DHW bottom-layer capacity C_bot [kWh/K].",
    )
    parser.add_argument(
        "--dhw-max-layer-spread-c",
        type=float,
        default=DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C,
        help="Maximum allowed |T_top − T_bot| for a standby-mixed calibration sample [°C].",
    )
    parser.add_argument(
        "--dhw-r-loss",
        type=float,
        default=None,
        help="Reference DHW standby-loss resistance R_loss [K/kW] used by active DHW calibration.",
    )
    parser.add_argument(
        "--dhw-r-strat",
        type=float,
        default=None,
        help="Reference DHW stratification resistance R_strat [K/kW] used as the active-fit prior.",
    )
    parser.add_argument(
        "--dhw-min-power-kw",
        type=float,
        default=DEFAULT_MIN_DHW_POWER_KW,
        help="Minimum mean DHW thermal charging power for an active DHW sample [kW].",
    )
    parser.add_argument(
        "--dhw-min-layer-spread-c",
        type=float,
        default=DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C,
        help="Minimum max(|T_top − T_bot|) required within an active DHW sample [°C].",
    )
    parser.add_argument(
        "--dhw-max-implied-tap-m3-per-h",
        type=float,
        default=DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H,
        help="Maximum implied tap flow allowed for active DHW no-draw samples [m³/h].",
    )
    parser.add_argument(
        "--dhw-min-segment-samples",
        type=int,
        default=DEFAULT_MIN_DHW_SEGMENT_SAMPLES,
        help="Minimum number of replay samples per contiguous active DHW run [-].",
    )
    parser.add_argument(
        "--cop-min-thermal-energy-kwh",
        type=float,
        default=DEFAULT_MIN_THERMAL_ENERGY_KWH,
        help="Minimum thermal energy required in a bucket before using it for COP calibration [kWh].",
    )
    parser.add_argument(
        "--cop-min-electric-energy-kwh",
        type=float,
        default=DEFAULT_MIN_ELECTRIC_ENERGY_KWH,
        help="Minimum electrical energy required in a bucket before using it for COP calibration [kWh].",
    )
    parser.add_argument(
        "--cop-min-ufh-samples",
        type=int,
        default=DEFAULT_MIN_UFH_CURVE_SAMPLE_COUNT,
        help="Minimum UFH buckets required to fit the heating curve in the COP stage [-].",
    )
    parser.add_argument(
        "--cop-min-segment-samples",
        type=int,
        default=DEFAULT_MIN_COP_SEGMENT_SAMPLES,
        help="Minimum contiguous same-mode buckets per raw COP segment before selection [-].",
    )
    parser.add_argument(
        "--cop-min-segment-thermal-energy-kwh",
        type=float,
        default=DEFAULT_MIN_COP_SEGMENT_THERMAL_ENERGY_KWH,
        help="Minimum delivered thermal energy required within a contiguous COP segment [kWh].",
    )
    parser.add_argument(
        "--cop-min-segment-actual-cop-span",
        type=float,
        default=DEFAULT_MIN_COP_SEGMENT_ACTUAL_COP_SPAN,
        help="Minimum measured COP span required within a contiguous COP segment [-].",
    )
    parser.add_argument(
        "--cop-max-segment-supply-tracking-rmse-c",
        type=float,
        default=DEFAULT_MAX_COP_SEGMENT_SUPPLY_TRACKING_RMSE_C,
        help="Maximum RMSE between measured and target supply temperatures within a COP segment [°C].",
    )
    parser.add_argument(
        "--cop-min-ufh-segment-outdoor-span-c",
        type=float,
        default=DEFAULT_MIN_UFH_COP_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
        help="Minimum outdoor-temperature span required within a UFH COP segment [°C].",
    )
    parser.add_argument(
        "--cop-min-ufh-segment-supply-target-span-c",
        type=float,
        default=DEFAULT_MIN_UFH_COP_SEGMENT_SUPPLY_TARGET_SPAN_C,
        help="Minimum target-supply-temperature span required within a UFH COP segment [°C].",
    )
    parser.add_argument(
        "--cop-min-segment-score",
        type=float,
        default=DEFAULT_MIN_COP_SEGMENT_SCORE,
        help="Minimum dimensionless quality score required for a COP segment to survive selection [-].",
    )
    parser.add_argument(
        "--cop-max-selected-ufh-segments",
        type=int,
        default=None,
        help="Optional top-N cap on selected UFH COP segments after quality ranking [-].",
    )
    parser.add_argument(
        "--cop-max-selected-dhw-segments",
        type=int,
        default=None,
        help="Optional top-N cap on selected DHW COP segments after quality ranking [-].",
    )
    parser.add_argument(
        "--cop-t-ref-outdoor-c",
        type=float,
        default=DEFAULT_T_REF_OUTDOOR_C,
        help="Fixed heating-curve reference outdoor temperature used by the COP stage [°C].",
    )
    parser.add_argument(
        "--cop-delta-t-cond-k",
        type=float,
        default=DEFAULT_DELTA_T_COND_K,
        help="Fixed condenser approach temperature used by the COP stage [K].",
    )
    parser.add_argument(
        "--cop-delta-t-evap-k",
        type=float,
        default=DEFAULT_DELTA_T_EVAP_K,
        help="Fixed evaporator approach temperature used by the COP stage [K].",
    )
    parser.add_argument(
        "--cop-min",
        type=float,
        default=DEFAULT_COP_MIN,
        help="Physical lower COP bound enforced during COP calibration [-].",
    )
    parser.add_argument(
        "--cop-max",
        type=float,
        default=DEFAULT_COP_MAX,
        help="Physical upper COP bound enforced during COP calibration [-].",
    )
    parser.add_argument(
        "--cop-heating-curve-loss",
        choices=COP_LEAST_SQUARES_LOSS_CHOICES,
        default=DEFAULT_COP_HEATING_CURVE_LOSS_NAME,
        help="Robust least-squares loss used for heating-curve fitting.",
    )
    parser.add_argument(
        "--cop-eta-loss",
        choices=COP_LEAST_SQUARES_LOSS_CHOICES,
        default=DEFAULT_COP_ETA_LOSS_NAME,
        help="Robust least-squares loss used for eta fitting against electrical energy.",
    )
    parser.add_argument(
        "--cop-heating-curve-loss-scale-c",
        type=float,
        default=DEFAULT_COP_HEATING_CURVE_LOSS_SCALE_C,
        help="Residual scale used by the chosen heating-curve robust loss [°C].",
    )
    parser.add_argument(
        "--cop-eta-loss-scale-kwh",
        type=float,
        default=DEFAULT_COP_ETA_LOSS_SCALE_KWH,
        help="Residual scale used by the chosen eta-fit robust loss [kWh].",
    )
    parser.add_argument(
        "--dhw-min-segment-delivered-energy-kwh",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH,
        help="Minimum delivered DHW charging energy within an active-DHW segment [kWh].",
    )
    parser.add_argument(
        "--dhw-min-segment-mean-layer-spread-c",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C,
        help="Minimum mean |T_top − T_bot| required within an active-DHW segment [°C].",
    )
    parser.add_argument(
        "--dhw-min-segment-layer-spread-span-c",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C,
        help="Minimum range of |T_top − T_bot| required within an active-DHW segment [°C].",
    )
    parser.add_argument(
        "--dhw-min-segment-bottom-rise-c",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C,
        help="Minimum net bottom-layer temperature rise required within an active-DHW segment [°C].",
    )
    parser.add_argument(
        "--dhw-min-segment-top-rise-c",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C,
        help="Minimum net top-layer temperature rise required within an active-DHW segment [°C].",
    )
    parser.add_argument(
        "--dhw-min-segment-score",
        type=float,
        default=DEFAULT_MIN_DHW_SEGMENT_SCORE,
        help="Minimum dimensionless quality score required for an active-DHW segment to be selected [-].",
    )
    parser.add_argument(
        "--dhw-max-selected-segments",
        type=int,
        default=None,
        help="Optional cap on the number of retained active-DHW segments; best-scoring runs are kept.",
    )
    parser.add_argument(
        "--fit-c-r",
        action="store_true",
        help="Also fit C_r during the active UFH stage instead of keeping it fixed.",
    )
    parser.add_argument(
        "--fit-initial-floor-offset",
        action="store_true",
        help="Fit a global initial floor-temperature offset that is applied at every UFH segment start.",
    )
    parser.add_argument(
        "--initial-floor-offset-c",
        type=float,
        default=DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C,
        help="Initial/fixed floor-temperature offset relative to room temperature at segment starts [°C].",
    )
    parser.add_argument(
        "--min-ufh-power-kw",
        type=float,
        default=DEFAULT_MIN_UFH_POWER_KW,
        help="Minimum mean UFH thermal power for an active sample [kW].",
    )
    parser.add_argument(
        "--min-segment-samples",
        type=int,
        default=DEFAULT_MIN_SEGMENT_SAMPLES,
        help="Minimum number of replay samples per contiguous UFH run [-].",
    )
    parser.add_argument(
        "--min-segment-power-span-kw",
        type=float,
        default=DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW,
        help="Minimum UFH power span required within a segment [kW].",
    )
    parser.add_argument(
        "--min-segment-room-span-c",
        type=float,
        default=DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C,
        help="Minimum room-temperature span required within a segment [°C].",
    )
    parser.add_argument(
        "--min-segment-outdoor-span-c",
        type=float,
        default=DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
        help="Minimum outdoor-temperature span required within a segment [°C].",
    )
    parser.add_argument(
        "--min-segment-score",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SCORE,
        help="Minimum dimensionless quality score required for a segment to be selected [-].",
    )
    parser.add_argument(
        "--max-selected-segments",
        type=int,
        default=None,
        help="Optional cap on the number of retained UFH segments; best-scoring runs are kept.",
    )
    parser.add_argument(
        "--max-gti-w-per-m2",
        type=float,
        default=None,
        help=(
            "Optional GTI threshold [W/m²]. Defaults to the low-solar off-mode cutoff for "
            "stage=off and to a wide acceptance range for stage=active-ufh."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=UFHOffCalibrationSettings().min_sample_count,
        help="Minimum number of transition samples required before fitting [-].",
    )
    parser.add_argument(
        "--reference-c-eff-kwh-per-k",
        type=float,
        default=None,
        help=(
            "Optional reference effective capacity [kWh/K]. When provided, the CLI "
            "also reports a derived R_ro = tau_house / C_eff_ref."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the calibration result as JSON only.",
    )
    return parser


def _build_reference_parameters(args: argparse.Namespace) -> ThermalParameters:
    """Build the required reference UFH parameter object for active calibration."""
    required_names = ("dt_hours", "c_r", "c_b", "r_br", "r_ro", "alpha", "eta", "a_glass")
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            "Active UFH calibration requires explicit reference thermal parameters: "
            f"missing {joined}."
        )
    return ThermalParameters(
        dt_hours=float(args.dt_hours),
        C_r=float(args.c_r),
        C_b=float(args.c_b),
        R_br=float(args.r_br),
        R_ro=float(args.r_ro),
        alpha=float(args.alpha),
        eta=float(args.eta),
        A_glass=float(args.a_glass),
    )


def _build_dhw_standby_settings(args: argparse.Namespace) -> DHWStandbyCalibrationSettings:
    """Build the required settings object for DHW standby-loss calibration."""
    required_names = ("dhw_dt_hours", "dhw_c_top", "dhw_c_bot")
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            "DHW standby calibration requires explicit reference capacities and time step: "
            f"missing {joined}."
        )
    return DHWStandbyCalibrationSettings(
        dt_hours=float(args.dhw_dt_hours),
        reference_c_top_kwh_per_k=float(args.dhw_c_top),
        reference_c_bot_kwh_per_k=float(args.dhw_c_bot),
        min_sample_count=args.min_samples,
        max_layer_temperature_spread_c=float(args.dhw_max_layer_spread_c),
    )


def _build_dhw_reference_parameters(args: argparse.Namespace) -> DHWParameters:
    """Build the required DHW parameter object for active stratification calibration."""
    required_names = ("dhw_dt_hours", "dhw_c_top", "dhw_c_bot", "dhw_r_loss", "dhw_r_strat")
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            "Active DHW calibration requires explicit reference DHW parameters: "
            f"missing {joined}."
        )
    return DHWParameters(
        dt_hours=float(args.dhw_dt_hours),
        C_top=float(args.dhw_c_top),
        C_bot=float(args.dhw_c_bot),
        R_strat=float(args.dhw_r_strat),
        R_loss=float(args.dhw_r_loss),
    )


def _build_cop_settings(args: argparse.Namespace) -> COPCalibrationSettings:
    """Build the settings object for offline COP calibration."""
    return COPCalibrationSettings(
        min_sample_count=args.min_samples,
        min_ufh_curve_sample_count=args.cop_min_ufh_samples,
        min_thermal_energy_kwh=args.cop_min_thermal_energy_kwh,
        min_electric_energy_kwh=args.cop_min_electric_energy_kwh,
        min_segment_samples=args.cop_min_segment_samples,
        min_segment_thermal_energy_kwh=args.cop_min_segment_thermal_energy_kwh,
        min_segment_actual_cop_span=args.cop_min_segment_actual_cop_span,
        max_segment_supply_tracking_rmse_c=args.cop_max_segment_supply_tracking_rmse_c,
        min_ufh_segment_outdoor_temperature_span_c=args.cop_min_ufh_segment_outdoor_span_c,
        min_ufh_segment_supply_target_span_c=args.cop_min_ufh_segment_supply_target_span_c,
        min_segment_score=args.cop_min_segment_score,
        max_selected_ufh_segments=args.cop_max_selected_ufh_segments,
        max_selected_dhw_segments=args.cop_max_selected_dhw_segments,
        t_ref_outdoor_c=args.cop_t_ref_outdoor_c,
        delta_t_cond_k=args.cop_delta_t_cond_k,
        delta_t_evap_k=args.cop_delta_t_evap_k,
        cop_min=args.cop_min,
        cop_max=args.cop_max,
        heating_curve_loss_name=args.cop_heating_curve_loss,
        eta_loss_name=args.cop_eta_loss,
        heating_curve_loss_scale_c=args.cop_heating_curve_loss_scale_c,
        eta_loss_scale_kwh=args.cop_eta_loss_scale_kwh,
    )


def main() -> None:
    """Run the requested offline calibration stage from telemetry history."""
    parser = _build_parser()
    args = parser.parse_args()
    repository = TelemetryRepository(database_url=args.database_url)

    if args.stage == "off":
        max_gti_w_per_m2 = (
            UFHOffCalibrationSettings().max_gti_w_per_m2
            if args.max_gti_w_per_m2 is None
            else args.max_gti_w_per_m2
        )
        settings = UFHOffCalibrationSettings(
            max_gti_w_per_m2=max_gti_w_per_m2,
            min_sample_count=args.min_samples,
            reference_c_eff_kwh_per_k=args.reference_c_eff_kwh_per_k,
        )
        dataset = build_ufh_off_dataset_from_repository(repository, settings)
        result = calibrate_ufh_off_from_repository(repository, settings)
        payload = {
            "stage": args.stage,
            "dataset": {
                "sample_count": dataset.sample_count,
                "start_utc": dataset.start_utc.isoformat(),
                "end_utc": dataset.end_utc.isoformat(),
            },
            "fit": {
                **asdict(result),
                "dataset_start_utc": result.dataset_start_utc.isoformat(),
                "dataset_end_utc": result.dataset_end_utc.isoformat(),
            },
        }
    elif args.stage == "cop":
        settings = _build_cop_settings(args)
        dataset = build_cop_dataset_from_repository(repository, settings)
        result = calibrate_cop_from_repository(repository, settings)
        payload = {
            "stage": args.stage,
            "dataset": {
                "sample_count": dataset.sample_count,
                "ufh_sample_count": dataset.ufh_sample_count,
                "dhw_sample_count": dataset.dhw_sample_count,
                "selected_segment_count": dataset.selected_segment_count,
                "selected_ufh_segment_count": dataset.selected_ufh_segment_count,
                "selected_dhw_segment_count": dataset.selected_dhw_segment_count,
                "raw_segment_count": dataset.raw_segment_count,
                "dropped_segment_count": dataset.dropped_segment_count,
                "segment_qualities": [asdict(quality) for quality in dataset.segment_qualities],
                "start_utc": dataset.start_utc.isoformat(),
                "end_utc": dataset.end_utc.isoformat(),
            },
            "fit": {
                **asdict(result),
                "dataset_start_utc": result.dataset_start_utc.isoformat(),
                "dataset_end_utc": result.dataset_end_utc.isoformat(),
            },
        }
    elif args.stage == "active-ufh":
        max_gti_w_per_m2 = (
            DEFAULT_ACTIVE_MAX_GTI_W_PER_M2 if args.max_gti_w_per_m2 is None else args.max_gti_w_per_m2
        )
        settings = UFHActiveCalibrationSettings(
            reference_parameters=_build_reference_parameters(args),
            max_gti_w_per_m2=max_gti_w_per_m2,
            min_sample_count=args.min_samples,
            min_segment_samples=args.min_segment_samples,
            min_segment_ufh_power_span_kw=args.min_segment_power_span_kw,
            min_segment_room_temperature_span_c=args.min_segment_room_span_c,
            min_segment_outdoor_temperature_span_c=args.min_segment_outdoor_span_c,
            min_segment_score=args.min_segment_score,
            max_selected_segments=args.max_selected_segments,
            min_ufh_power_kw=args.min_ufh_power_kw,
            fit_c_r=bool(args.fit_c_r),
            fit_initial_floor_temperature_offset=bool(args.fit_initial_floor_offset),
            initial_floor_temperature_offset_c=args.initial_floor_offset_c,
        )
        dataset = build_ufh_active_dataset_from_repository(repository, settings)
        result = calibrate_ufh_active_from_repository(repository, settings)
        payload = {
            "stage": args.stage,
            "dataset": {
                "sample_count": dataset.sample_count,
                "segment_count": dataset.segment_count,
                "raw_segment_count": dataset.raw_segment_count,
                "dropped_segment_count": dataset.dropped_segment_count,
                "start_utc": dataset.start_utc.isoformat(),
                "end_utc": dataset.end_utc.isoformat(),
            },
            "fit": {
                **asdict(result),
                "dataset_start_utc": result.dataset_start_utc.isoformat(),
                "dataset_end_utc": result.dataset_end_utc.isoformat(),
            },
        }
    elif args.stage == "active-dhw":
        settings = DHWActiveCalibrationSettings(
            reference_parameters=_build_dhw_reference_parameters(args),
            min_sample_count=args.min_samples,
            min_segment_samples=args.dhw_min_segment_samples,
            min_dhw_power_kw=args.dhw_min_power_kw,
            min_layer_temperature_spread_c=args.dhw_min_layer_spread_c,
            max_implied_tap_m3_per_h=args.dhw_max_implied_tap_m3_per_h,
            min_segment_delivered_energy_kwh=args.dhw_min_segment_delivered_energy_kwh,
            min_segment_mean_layer_spread_c=args.dhw_min_segment_mean_layer_spread_c,
            min_segment_layer_spread_span_c=args.dhw_min_segment_layer_spread_span_c,
            min_segment_bottom_temperature_rise_c=args.dhw_min_segment_bottom_rise_c,
            min_segment_top_temperature_rise_c=args.dhw_min_segment_top_rise_c,
            min_segment_score=args.dhw_min_segment_score,
            max_selected_segments=args.dhw_max_selected_segments,
        )
        dataset = build_dhw_active_dataset_from_repository(repository, settings)
        result = calibrate_dhw_active_from_repository(repository, settings)
        payload = {
            "stage": args.stage,
            "dataset": {
                "sample_count": dataset.sample_count,
                "segment_count": dataset.segment_count,
                "raw_segment_count": dataset.raw_segment_count,
                "dropped_segment_count": dataset.dropped_segment_count,
                "start_utc": dataset.start_utc.isoformat(),
                "end_utc": dataset.end_utc.isoformat(),
            },
            "fit": {
                **asdict(result),
                "dataset_start_utc": result.dataset_start_utc.isoformat(),
                "dataset_end_utc": result.dataset_end_utc.isoformat(),
            },
        }
    else:
        settings = _build_dhw_standby_settings(args)
        dataset = build_dhw_standby_dataset_from_repository(repository, settings)
        result = calibrate_dhw_standby_from_repository(repository, settings)
        payload = {
            "stage": args.stage,
            "dataset": {
                "sample_count": dataset.sample_count,
                "start_utc": dataset.start_utc.isoformat(),
                "end_utc": dataset.end_utc.isoformat(),
            },
            "fit": {
                **asdict(result),
                "dataset_start_utc": result.dataset_start_utc.isoformat(),
                "dataset_end_utc": result.dataset_end_utc.isoformat(),
            },
        }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.stage == "off":
        print("UFH off-mode calibration")
        print("========================")
        print(f"Samples          : {dataset.sample_count}")
        print(f"Window           : {dataset.start_utc.isoformat()} -> {dataset.end_utc.isoformat()}")
        print(f"tau_house        : {result.tau_house_hours:.4f} h")
        if result.suggested_r_ro_k_per_kw is not None:
            print(f"R_ro (derived)   : {result.suggested_r_ro_k_per_kw:.4f} K/kW")
        print(f"RMSE(T_room)     : {result.rmse_room_temperature_c:.5f} °C")
        print(f"Max |residual|   : {result.max_abs_residual_c:.5f} °C")
        print(f"Optimizer status : {result.optimizer_status}")
        return

    if args.stage == "cop":
        print("COP calibration")
        print("===============")
        print(f"Samples              : {dataset.sample_count}")
        print(f"UFH samples          : {dataset.ufh_sample_count}")
        print(f"DHW samples          : {dataset.dhw_sample_count}")
        print(f"Selected segments    : {dataset.selected_segment_count}")
        print(f"Selected UFH segs    : {dataset.selected_ufh_segment_count}")
        print(f"Selected DHW segs    : {dataset.selected_dhw_segment_count}")
        print(f"Raw segments         : {dataset.raw_segment_count}")
        print(f"Dropped segments     : {dataset.dropped_segment_count}")
        print(f"Window               : {dataset.start_utc.isoformat()} -> {dataset.end_utc.isoformat()}")
        print(f"eta_carnot           : {result.fitted_parameters.eta_carnot:.6f} [-]")
        print(f"T_supply_min         : {result.fitted_parameters.T_supply_min:.6f} °C")
        print(f"heating_curve_slope  : {result.fitted_parameters.heating_curve_slope:.6f} K/K")
        print(f"Heating-curve loss   : {settings.heating_curve_loss_name}")
        print(f"Eta-fit loss         : {settings.eta_loss_name}")
        print(f"RMSE(T_supply)       : {result.rmse_supply_temperature_c:.5f} °C")
        print(f"RMSE(E_elec)         : {result.rmse_electric_energy_kwh:.5f} kWh")
        print(f"RMSE(COP)            : {result.rmse_actual_cop:.5f} [-]")
        print(f"UFH RMSE(E_elec)     : {result.ufh_rmse_electric_energy_kwh:.5f} kWh")
        if result.dhw_rmse_electric_energy_kwh is not None:
            print(f"DHW RMSE(E_elec)     : {result.dhw_rmse_electric_energy_kwh:.5f} kWh")
        print(f"UFH RMSE(COP)        : {result.ufh_rmse_actual_cop:.5f} [-]")
        if result.dhw_rmse_actual_cop is not None:
            print(f"DHW RMSE(COP)        : {result.dhw_rmse_actual_cop:.5f} [-]")
        print(f"UFH bias(COP)        : {result.ufh_bias_actual_cop:.5f} [-]")
        if result.dhw_bias_actual_cop is not None:
            print(f"DHW bias(COP)        : {result.dhw_bias_actual_cop:.5f} [-]")
        print(f"UFH diagnostic eta   : {result.diagnostic_eta_carnot_ufh:.6f} [-]")
        if result.diagnostic_eta_carnot_dhw is not None:
            print(f"DHW diagnostic eta   : {result.diagnostic_eta_carnot_dhw:.6f} [-]")
        print(f"Heating-curve status : {result.heating_curve_optimizer_status}")
        print(f"Eta status           : {result.eta_optimizer_status}")
        return

    if args.stage == "dhw-standby":
        print("DHW standby calibration")
        print("=======================")
        print(f"Samples          : {dataset.sample_count}")
        print(f"Window           : {dataset.start_utc.isoformat()} -> {dataset.end_utc.isoformat()}")
        print(f"tau_standby      : {result.tau_standby_hours:.4f} h")
        print(f"R_loss (derived) : {result.suggested_r_loss_k_per_kw:.4f} K/kW")
        print(f"RMSE(T_dhw)      : {result.rmse_mean_tank_temperature_c:.5f} °C")
        print(f"Max |residual|   : {result.max_abs_residual_c:.5f} °C")
        print(f"Optimizer status : {result.optimizer_status}")
        return

    if args.stage == "active-dhw":
        print("DHW active stratification calibration")
        print("===================================")
        print(f"Samples          : {dataset.sample_count}")
        print(f"Segments         : {dataset.segment_count}")
        print(f"Raw segments     : {dataset.raw_segment_count}")
        print(f"Dropped segments : {dataset.dropped_segment_count}")
        print(f"Window           : {dataset.start_utc.isoformat()} -> {dataset.end_utc.isoformat()}")
        print(f"R_strat          : {result.fitted_parameters.R_strat:.6f} K/kW")
        print(f"R_loss (fixed)   : {result.fitted_parameters.R_loss:.6f} K/kW")
        print(f"RMSE(T_top)      : {result.rmse_t_top_c:.5f} °C")
        print(f"RMSE(T_bot)      : {result.rmse_t_bot_c:.5f} °C")
        print(f"Max |residual|   : {result.max_abs_residual_c:.5f} °C")
        print(f"Optimizer status : {result.optimizer_status}")
        return

    print("UFH active RC calibration")
    print("=========================")
    print(f"Samples          : {dataset.sample_count}")
    print(f"Segments         : {dataset.segment_count}")
    print(f"Raw segments     : {dataset.raw_segment_count}")
    print(f"Dropped segments : {dataset.dropped_segment_count}")
    print(f"Window           : {dataset.start_utc.isoformat()} -> {dataset.end_utc.isoformat()}")
    print(f"fit C_r          : {result.fit_c_r}")
    print(f"fit T_b offset   : {result.fit_initial_floor_temperature_offset}")
    print(f"C_r              : {result.fitted_parameters.C_r:.6f} kWh/K")
    print(f"C_b              : {result.fitted_parameters.C_b:.6f} kWh/K")
    print(f"R_br             : {result.fitted_parameters.R_br:.6f} K/kW")
    print(f"R_ro             : {result.fitted_parameters.R_ro:.6f} K/kW")
    print(f"T_b offset       : {result.fitted_initial_floor_temperature_offset_c:.6f} °C")
    print(f"RMSE(T_room)     : {result.rmse_room_temperature_c:.5f} °C")
    print(f"Max |innovation| : {result.max_abs_innovation_c:.5f} °C")
    print(f"Optimizer status : {result.optimizer_status}")


if __name__ == "__main__":
    main()

