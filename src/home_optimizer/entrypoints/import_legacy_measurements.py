from __future__ import annotations

import argparse
from pathlib import Path

from home_optimizer.infrastructure.database.legacy_measurement_importer import (
    DEFAULT_LEGACY_MEASUREMENT_SOURCE,
    import_legacy_measurements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import legacy 15-minute measurement rows into samples_15m.",
    )
    parser.add_argument(
        "--legacy-db",
        default="database-old.sqlite",
        help="Path to the legacy SQLite database containing the measurement table.",
    )
    parser.add_argument(
        "--target-db",
        default="database.sqlite",
        help="Path to the target Home Optimizer SQLite database.",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone to apply to naive legacy timestamps before converting to UTC.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_LEGACY_MEASUREMENT_SOURCE,
        help="Source label stored in samples_15m.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for the selected source in samples_15m before importing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and map the legacy rows without writing samples to the target database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_legacy_measurements(
        legacy_db_path=str(Path(args.legacy_db).expanduser()),
        target_db_path=str(Path(args.target_db).expanduser()),
        timezone_name=args.timezone,
        source=args.source,
        replace=args.replace,
        dry_run=args.dry_run,
    )
    action = "would write" if args.dry_run else "wrote"
    source_parts = [f"{summary.measurement_rows} measurement rows"]
    if summary.solar_forecast_rows:
        source_parts.append(f"{summary.solar_forecast_rows} solar_forecast rows")
    print(
        f"Imported {', '.join(source_parts)}; "
        f"generated {summary.generated_samples_15m} samples_15m rows and {action} "
        f"{summary.written_samples_15m} rows into {args.target_db}."
    )


if __name__ == "__main__":
    main()

