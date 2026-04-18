"""Command-line entry point for offline thermal calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import (
    DEFAULT_ACTIVE_MAX_GTI_W_PER_M2,
    DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C,
    DEFAULT_MIN_UFH_POWER_KW,
    DEFAULT_MIN_SEGMENT_SAMPLES,
    UFHActiveCalibrationSettings,
    UFHOffCalibrationSettings,
)
from .service import (
    build_ufh_active_dataset_from_repository,
    build_ufh_off_dataset_from_repository,
    calibrate_ufh_active_from_repository,
    calibrate_ufh_off_from_repository,
)
from ..telemetry.repository import TelemetryRepository
from ..types import ThermalParameters

DEFAULT_DATABASE_URL: str = "sqlite:///database.sqlite3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="home-optimizer-calibration",
        description="Fit first-stage thermal parameters from persisted telemetry history.",
    )
    parser.add_argument(
        "--stage",
        choices=("off", "active-ufh"),
        default="off",
        help="Calibration stage to run: passive off-mode envelope fit or active UFH RC fit.",
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
    else:
        max_gti_w_per_m2 = (
            DEFAULT_ACTIVE_MAX_GTI_W_PER_M2 if args.max_gti_w_per_m2 is None else args.max_gti_w_per_m2
        )
        settings = UFHActiveCalibrationSettings(
            reference_parameters=_build_reference_parameters(args),
            max_gti_w_per_m2=max_gti_w_per_m2,
            min_sample_count=args.min_samples,
            min_segment_samples=args.min_segment_samples,
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

    print("UFH active RC calibration")
    print("=========================")
    print(f"Samples          : {dataset.sample_count}")
    print(f"Segments         : {dataset.segment_count}")
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

