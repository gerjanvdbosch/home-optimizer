from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

from home_optimizer.app.container_factories import build_local_container
from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings_loader import load_settings
from home_optimizer.features.mpc.control_oriented import StateSpaceActuatorSensitivityService

LOGGER = logging.getLogger(__name__)


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO datetime: {value}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include timezone offset")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect actuator sensitivity for different thermostat setpoints from the same initial state.",
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
        "--start-time",
        required=True,
        type=parse_datetime,
        help="Inclusive ISO timestamp with timezone, e.g. 2026-04-27T10:00:00+00:00.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=6,
        help="Inspection horizon in hours.",
    )
    parser.add_argument("--setpoint-min", type=float, required=True, help="Minimum setpoint to inspect.")
    parser.add_argument("--setpoint-max", type=float, required=True, help="Maximum setpoint to inspect.")
    parser.add_argument("--setpoint-step", type=float, default=0.5, help="Setpoint increment.")
    parser.add_argument(
        "--model-name",
        default="linear_2state_room_temperature",
        help="Stored room model name to use.",
    )
    return parser.parse_args()


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_setpoints(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("setpoint-step must be > 0")
    if minimum > maximum:
        raise ValueError("setpoint-min must be <= setpoint-max")
    values: list[float] = []
    current = minimum
    while current <= maximum + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, overrides=args.set)
    configure_logging(settings.log_level)

    end_time = args.start_time + timedelta(hours=args.horizon_hours)
    LOGGER.info(
        "Starting actuator sensitivity inspection at %s for %s h",
        args.start_time,
        args.horizon_hours,
    )
    container = build_local_container(settings)
    try:
        service = StateSpaceActuatorSensitivityService(
            prediction_service=container.prediction_service,
        )
        result = service.inspect(
            start_time=args.start_time,
            end_time=end_time,
            setpoints=build_setpoints(args.setpoint_min, args.setpoint_max, args.setpoint_step),
            model_name=args.model_name,
        )
    finally:
        container.close()

    print(
        f"Model: {result.model_name} | thermal model: {result.thermal_output_model_name} | "
        f"interval: {result.interval_minutes} min"
    )
    print(
        f"Window: {result.start_time.isoformat()} -> {result.end_time.isoformat()} | "
        f"initial room={result.initial_room_temperature:.3f} | "
        f"initial floor={result.initial_floor_heat_state:.3f} | "
        f"initial thermal={result.initial_thermal_output:.3f} | "
        f"first supply_tgt={format_metric(result.first_supply_target_temperature, 2)}"
    )
    print("")
    print("setpt demand first_th peak_th avg_th end_room max_room")
    for row in result.rows:
        print(
            f"{row.thermostat_setpoint:>5.2f} "
            f"{row.initial_heating_demand:>6.3f} "
            f"{format_metric(row.first_predicted_thermal_output):>8} "
            f"{format_metric(row.peak_predicted_thermal_output):>7} "
            f"{format_metric(row.average_predicted_thermal_output):>6} "
            f"{format_metric(row.final_room_temperature):>8} "
            f"{format_metric(row.maximum_room_temperature):>8}"
        )


if __name__ == "__main__":
    main()
