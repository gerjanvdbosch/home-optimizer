from __future__ import annotations

import argparse
import logging
from datetime import date

from home_optimizer.app.container_factories import build_local_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.features.backtesting import ThermalOutputBacktestingService

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local thermal_output backtest against historical database data.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to local app config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config using dot notation, e.g. --set database_path=database.sqlite.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=date.fromisoformat,
        help="First local day to backtest, e.g. 2026-04-01.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=date.fromisoformat,
        help="Last local day to backtest, e.g. 2026-04-30.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Prediction horizon per day in hours, e.g. 6, 12, or 24.",
    )
    parser.add_argument(
        "--model-name",
        default="linear_1step_thermal_output",
        help="Stored model name to backtest, e.g. linear_1step_thermal_output.",
    )
    parser.add_argument(
        "--details-day",
        type=date.fromisoformat,
        default=None,
        help="Optional local day to print detailed aligned thermal_output rows for.",
    )
    return parser.parse_args()


def build_backtesting_service(
    *,
    config_path: str,
    overrides: list[str],
) -> tuple[object, ThermalOutputBacktestingService]:
    settings = load_settings(config_path, overrides=overrides)
    container = build_local_container(settings)
    return container, container.thermal_output_backtesting_service


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, overrides=args.set)
    configure_logging(settings.log_level)

    LOGGER.info(
        "Starting thermal_output backtest for %s to %s",
        args.start_date,
        args.end_date,
    )
    container, service = build_backtesting_service(
        config_path=args.config,
        overrides=args.set,
    )
    try:
        result = service.backtest_by_day(
            start_date=args.start_date,
            end_date=args.end_date,
            horizon_hours=args.horizon_hours,
            model_name=args.model_name,
        )
    finally:
        container.close()

    print(
        f"Model: {result.model_name} | interval: {result.interval_minutes} min | "
        f"horizon: {result.horizon_hours} h | "
        f"days: {result.total_days} | success: {result.successful_days} | failed: {result.failed_days}"
    )
    print(
        f"Average RMSE: {format_metric(result.average_rmse)} | "
        f"Average bias: {format_metric(result.average_bias)} | "
        f"Average max error: {format_metric(result.average_max_absolute_error)}"
    )
    print(f"Worst day by RMSE: {result.worst_day_by_rmse or '-'}")
    print("")
    print(
        "day         overlap  rmse    bias    max_err min_act max_act min_pred max_pred status"
    )
    for day_result in result.day_results:
        status = day_result.error or "ok"
        print(
            f"{day_result.day} "
            f"{day_result.overlap_count:>7} "
            f"{format_metric(day_result.rmse):>7} "
            f"{format_metric(day_result.bias):>7} "
            f"{format_metric(day_result.max_absolute_error):>7} "
            f"{format_metric(day_result.minimum_actual_thermal_output):>7} "
            f"{format_metric(day_result.maximum_actual_thermal_output):>7} "
            f"{format_metric(day_result.minimum_predicted_thermal_output):>8} "
            f"{format_metric(day_result.maximum_predicted_thermal_output):>8} "
            f"{status}"
        )

    if args.details_day is not None:
        matching_day = next(
            (day_result for day_result in result.day_results if day_result.day == args.details_day),
            None,
        )
        if matching_day is None:
            print("")
            print(f"No day result found for {args.details_day}.")
            return
        if matching_day.error is not None:
            print("")
            print(f"Cannot print details for {args.details_day}: {matching_day.error}")
            return

        print("")
        print(f"Details for {args.details_day}:")
        print("timestamp                  actual   pred room_t setpt demand supply_tgt")
        for point in matching_day.diagnostic_points:
            print(
                f"{point.timestamp} "
                f"{point.actual_thermal_output:>7.3f} "
                f"{point.predicted_thermal_output:>6.3f} "
                f"{point.room_temperature:>6.2f} "
                f"{point.thermostat_setpoint:>5.2f} "
                f"{point.heating_demand:>6.2f} "
                f"{point.supply_target_temperature:>10.2f}"
            )


if __name__ == "__main__":
    main()
