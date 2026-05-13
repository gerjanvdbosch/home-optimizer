from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.time import current_local_timezone
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcHorizonStep,
    MpcInitialState,
    MpcPlan,
)


def explain_heating_plan(
    *,
    plan: MpcPlan,
    control_model: LinearThermalControlModel,
    initial_state: MpcInitialState,
    horizon: list[MpcHorizonStep],
) -> str | None:
    if not plan.steps:
        return None

    first_heating_step = next((step for step in plan.steps if step.hp_on), None)
    if first_heating_step is None:
        return "No heating scheduled; comfort-min stays satisfied without heat."

    no_heat_rollout = rollout_without_heating(
        control_model=control_model,
        initial_state=initial_state,
        horizon=horizon,
    )
    first_no_heat_violation = next(
        (
            (step, temp_c)
            for step, temp_c in zip(horizon, no_heat_rollout, strict=True)
            if temp_c < step.temp_min_c
        ),
        None,
    )
    if first_no_heat_violation is None:
        return (
            "Heating is scheduled even though no comfort-min violation is forecast without heat."
        )

    violation_step, violation_temp_c = first_no_heat_violation
    return (
        "Heating scheduled to prevent comfort-min violation at "
        f"{_format_local_time(violation_step.timestamp_utc)}. "
        "No-heat rollout reaches "
        f"{violation_temp_c:.1f} °C vs min {violation_step.temp_min_c:.1f} °C."
    )


def rollout_without_heating(
    *,
    control_model: LinearThermalControlModel,
    initial_state: MpcInitialState,
    horizon: list[MpcHorizonStep],
) -> list[float]:
    if not horizon:
        return []

    predicted_temperatures = [initial_state.room_temp_c]
    current_temp_c = initial_state.room_temp_c
    for step in horizon[:-1]:
        current_temp_c = control_model.predict_next_temperature(
            room_temp_c=current_temp_c,
            outdoor_temp_c=step.outdoor_temp_c,
            solar_gain_kw=step.solar_gain_kw,
            heating_effect_kw=0.0,
            occupied=step.occupied,
        )
        predicted_temperatures.append(current_temp_c)
    return predicted_temperatures


def _format_local_time(timestamp_utc: datetime) -> str:
    local_timestamp = timestamp_utc.astimezone(current_local_timezone())
    return local_timestamp.strftime("%H:%M")
