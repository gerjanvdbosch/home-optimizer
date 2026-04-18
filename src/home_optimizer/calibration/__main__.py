"""Command-line entry point for offline thermal calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import UFHOffCalibrationSettings
from .service import build_ufh_off_dataset_from_repository, calibrate_ufh_off_from_repository
from ..telemetry.repository import TelemetryRepository

DEFAULT_DATABASE_URL: str = "sqlite:///database.sqlite3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="home-optimizer-calibration",
        description="Fit first-stage thermal parameters from persisted telemetry history.",
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="SQLAlchemy database URL containing telemetry history.",
    )
    parser.add_argument(
        "--max-gti-w-per-m2",
        type=float,
        default=UFHOffCalibrationSettings().max_gti_w_per_m2,
        help="Upper GTI threshold for low-solar off-mode selection [W/m²].",
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


def main() -> None:
    """Run the first-stage offline UFH envelope calibration from telemetry history."""
    parser = _build_parser()
    args = parser.parse_args()

    settings = UFHOffCalibrationSettings(
        max_gti_w_per_m2=args.max_gti_w_per_m2,
        min_sample_count=args.min_samples,
        reference_c_eff_kwh_per_k=args.reference_c_eff_kwh_per_k,
    )
    repository = TelemetryRepository(database_url=args.database_url)
    dataset = build_ufh_off_dataset_from_repository(repository, settings)
    result = calibrate_ufh_off_from_repository(repository, settings)
    payload = {
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


if __name__ == "__main__":
    main()

