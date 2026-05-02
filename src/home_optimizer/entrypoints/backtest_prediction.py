from __future__ import annotations

import argparse
import logging
from datetime import date

from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.app.container_factories import build_local_container
from home_optimizer.features.backtesting import RoomTemperatureBacktestingService

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local room temperature prediction backtest against historical database data.",
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
        "--comfort-min",
        type=float,
        default=None,
        help="Optional minimum comfort temperature in degC.",
    )
    parser.add_argument(
        "--comfort-max",
        type=float,
        default=None,
        help="Optional maximum comfort temperature in degC.",
    )
    return parser.parse_args()


def build_backtesting_service(
    *,
    config_path: str,
    overrides: list[str],
) -> tuple[object, RoomTemperatureBacktestingService]:
    settings = load_settings(config_path, overrides=overrides)
    container = build_local_container(settings)
    return container, container.backtesting_service


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, overrides=args.set)
    configure_logging(settings.log_level)

    LOGGER.info(
        "Starting room temperature prediction backtest for %s to %s",
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
            comfort_min_temperature=args.comfort_min,
            comfort_max_temperature=args.comfort_max,
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
        "day         overlap  rmse    bias    max_err min_pred max_pred under over status"
    )
    for day_result in result.day_results:
        status = day_result.error or "ok"
        under = "-" if day_result.under_comfort_count is None else str(day_result.under_comfort_count)
        over = "-" if day_result.over_comfort_count is None else str(day_result.over_comfort_count)
        print(
            f"{day_result.day} "
            f"{day_result.overlap_count:>7} "
            f"{format_metric(day_result.rmse):>7} "
            f"{format_metric(day_result.bias):>7} "
            f"{format_metric(day_result.max_absolute_error):>7} "
            f"{format_metric(day_result.minimum_predicted_temperature, 2):>8} "
            f"{format_metric(day_result.maximum_predicted_temperature, 2):>8} "
            f"{under:>5} "
            f"{over:>4} "
            f"{status}"
        )


if __name__ == "__main__":
    main()
