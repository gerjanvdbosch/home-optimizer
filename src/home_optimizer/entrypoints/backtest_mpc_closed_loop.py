from __future__ import annotations

import argparse
import logging
from datetime import date

from home_optimizer.app.container_factories import build_local_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.features.mpc import (
    DEFAULT_MPC_HORIZON_HOURS,
    ThermostatSetpointMpcClosedLoopService,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local closed-loop MPC backtest with receding-horizon replanning.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to local app config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config using dot notation, e.g. --set database_path=database.sqlite.",
    )
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=DEFAULT_MPC_HORIZON_HOURS,
        help="Receding-horizon planning window in hours.",
    )
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--switch-hours", type=int, default=2)
    parser.add_argument("--setpoint-min", type=float, default=19.0)
    parser.add_argument("--setpoint-max", type=float, default=21.0)
    parser.add_argument("--setpoint-step", type=float, default=0.5)
    parser.add_argument("--comfort-min", type=float, default=19.0)
    parser.add_argument("--comfort-max", type=float, default=21.0)
    parser.add_argument("--change-penalty", type=float, default=0.1)
    return parser.parse_args()


def build_allowed_setpoints(minimum: float, maximum: float, step: float) -> list[float]:
    if minimum > maximum:
        raise ValueError("setpoint-min must be <= setpoint-max")
    if step <= 0:
        raise ValueError("setpoint-step must be greater than zero")

    values: list[float] = []
    value = minimum
    while value <= maximum + 1e-9:
        values.append(round(value, 3))
        value += step
    return values


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, overrides=args.set)
    configure_logging(settings.log_level)
    LOGGER.info(
        "Starting closed-loop MPC backtest for %s to %s",
        args.start_date,
        args.end_date,
    )

    container = build_local_container(settings)
    try:
        service = ThermostatSetpointMpcClosedLoopService(
            container.time_series_read_repository,
            container.mpc_planner,
        )
        result = service.evaluate_by_day(
            start_date=args.start_date,
            end_date=args.end_date,
            allowed_setpoints=build_allowed_setpoints(
                args.setpoint_min,
                args.setpoint_max,
                args.setpoint_step,
            ),
            horizon_hours=args.horizon_hours,
            interval_minutes=args.interval_minutes,
            switch_interval_hours=args.switch_hours,
            comfort_min_temperature=args.comfort_min,
            comfort_max_temperature=args.comfort_max,
            setpoint_change_penalty=args.change_penalty,
        )
    finally:
        container.close()

    print(
        f"Model: {result.model_name or '-'} | interval: {result.interval_minutes} min | "
        f"horizon: {result.horizon_hours} h | days: {result.total_days} | "
        f"success: {result.successful_days} | failed: {result.failed_days}"
    )
    print(
        f"Average total cost: {result.average_total_cost:.3f}" if result.average_total_cost is not None
        else "Average total cost: -"
    )
    print(
        "Average comfort cost: "
        + (
            f"{result.average_comfort_violation_cost:.3f}"
            if result.average_comfort_violation_cost is not None
            else "-"
        )
        + " | Average switch cost: "
        + (
            f"{result.average_setpoint_change_cost:.3f}"
            if result.average_setpoint_change_cost is not None
            else "-"
        )
    )
    print("")
    print("day         avg_cost comfort switch min_pred max_pred under over status")
    for day_result in result.day_results:
        status = day_result.error or "ok"
        print(
            f"{day_result.day} "
            f"{day_result.average_total_cost:>8.3f} "
            f"{day_result.average_comfort_violation_cost:>7.3f} "
            f"{day_result.average_setpoint_change_cost:>6.3f} "
            f"{day_result.minimum_predicted_temperature if day_result.minimum_predicted_temperature is not None else '-':>8} "
            f"{day_result.maximum_predicted_temperature if day_result.maximum_predicted_temperature is not None else '-':>8} "
            f"{day_result.under_comfort_count:>5} "
            f"{day_result.over_comfort_count:>4} "
            f"{status}"
        )


if __name__ == "__main__":
    main()
