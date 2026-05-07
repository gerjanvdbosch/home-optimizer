from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from math import exp, log, sqrt

import numpy as np

from home_optimizer.domain.models import DomainModel
from home_optimizer.features.identification.models import IdentificationDataset, IdentificationDatasetRow


class RoomThermalModel(DomainModel):
    interval_minutes: int
    mass_decay_per_step: float
    intercept_c: float
    room_temperature_coefficient: float
    mass_temperature_coefficient: float
    outdoor_temperature_coefficient: float
    thermal_output_coefficient: float
    solar_gain_coefficient: float
    training_sample_count: int


class RoomThermalValidationMetric(DomainModel):
    horizon_minutes: int
    horizon_steps: int
    sample_count: int
    mae_c: float
    rmse_c: float
    bias_c: float


class RoomThermalValidationReport(DomainModel):
    total_valid_rows: int
    training_rows: int
    holdout_rows: int
    metrics: list[RoomThermalValidationMetric]


class RoomThermalModelFitResult(DomainModel):
    model: RoomThermalModel
    validation: RoomThermalValidationReport


class RoomThermalModelService:
    def fit_and_validate(
        self,
        dataset: IdentificationDataset,
        *,
        train_ratio: float = 0.7,
        mass_half_life_hours: float = 8.0,
    ) -> RoomThermalModelFitResult:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        if mass_half_life_hours <= 0:
            raise ValueError("mass_half_life_hours must be greater than zero")

        rows = _valid_room_rows(dataset)
        if len(rows) < 12:
            raise ValueError("at least 12 valid room-identification rows are required")

        split_index = int(len(rows) * train_ratio)
        split_index = max(6, min(split_index, len(rows) - 6))
        training_rows = rows[:split_index]
        holdout_rows = rows[split_index:]

        if len(training_rows) < 2:
            raise ValueError("at least 2 training rows are required")
        if len(holdout_rows) < 2:
            raise ValueError("at least 2 holdout rows are required")

        mass_decay = _mass_decay_per_step(
            interval_minutes=dataset.interval_minutes,
            mass_half_life_hours=mass_half_life_hours,
        )
        initial_mass = training_rows[0].room_temperature_c
        assert initial_mass is not None

        training_mass = _build_mass_state(
            rows=training_rows,
            mass_decay=mass_decay,
            initial_mass=initial_mass,
        )
        holdout_mass = _build_mass_state(
            rows=holdout_rows,
            mass_decay=mass_decay,
            initial_mass=training_mass[-1],
        )
        coefficients = _fit_room_equation(training_rows, training_mass)

        model = RoomThermalModel(
            interval_minutes=dataset.interval_minutes,
            mass_decay_per_step=mass_decay,
            intercept_c=float(coefficients[0]),
            room_temperature_coefficient=float(coefficients[1]),
            mass_temperature_coefficient=float(coefficients[2]),
            outdoor_temperature_coefficient=float(coefficients[3]),
            thermal_output_coefficient=float(coefficients[4]),
            solar_gain_coefficient=float(coefficients[5]),
            training_sample_count=len(training_rows) - 1,
        )

        validation = _validate_model(
            model=model,
            holdout_rows=holdout_rows,
            holdout_mass=holdout_mass,
            total_valid_rows=len(rows),
            training_rows=len(training_rows),
        )
        return RoomThermalModelFitResult(model=model, validation=validation)


def _valid_room_rows(dataset: IdentificationDataset) -> list[IdentificationDatasetRow]:
    expected_step = timedelta(minutes=dataset.interval_minutes)
    filtered = [
        row
        for row in dataset.rows
        if row.is_valid_for_room_identification
        and row.room_temperature_c is not None
        and row.outdoor_temperature_c is not None
        and row.thermal_output_estimate_kw is not None
        and row.solar_gain_proxy_w_m2 is not None
    ]
    if not filtered:
        return []

    contiguous_rows = [filtered[0]]
    for row in filtered[1:]:
        previous = contiguous_rows[-1]
        if row.timestamp_utc - previous.timestamp_utc != expected_step:
            continue
        contiguous_rows.append(row)
    return contiguous_rows


def _mass_decay_per_step(*, interval_minutes: int, mass_half_life_hours: float) -> float:
    interval_hours = interval_minutes / 60.0
    return exp(-log(2.0) * interval_hours / mass_half_life_hours)


def _build_mass_state(
    *,
    rows: Sequence[IdentificationDatasetRow],
    mass_decay: float,
    initial_mass: float,
) -> list[float]:
    mass_values = [initial_mass]
    current_mass = initial_mass
    for row in rows[1:]:
        room_temperature = row.room_temperature_c
        assert room_temperature is not None
        current_mass = mass_decay * current_mass + (1.0 - mass_decay) * room_temperature
        mass_values.append(current_mass)
    return mass_values


def _fit_room_equation(
    rows: Sequence[IdentificationDatasetRow],
    mass_values: Sequence[float],
) -> np.ndarray:
    if len(rows) != len(mass_values):
        raise ValueError("rows and mass_values must have the same length")
    if len(rows) < 2:
        raise ValueError("at least two rows are required to fit the model")

    design_rows: list[list[float]] = []
    targets: list[float] = []
    for index in range(len(rows) - 1):
        current = rows[index]
        next_row = rows[index + 1]
        assert current.room_temperature_c is not None
        assert current.outdoor_temperature_c is not None
        assert current.thermal_output_estimate_kw is not None
        assert current.solar_gain_proxy_w_m2 is not None
        assert next_row.room_temperature_c is not None

        design_rows.append(
            [
                1.0,
                current.room_temperature_c,
                mass_values[index],
                current.outdoor_temperature_c,
                current.thermal_output_estimate_kw,
                current.solar_gain_proxy_w_m2,
            ]
        )
        targets.append(next_row.room_temperature_c)

    design_matrix = np.array(design_rows, dtype=float)
    target_vector = np.array(targets, dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    return coefficients


def _predict_next_room_temperature(
    model: RoomThermalModel,
    *,
    room_temperature_c: float,
    mass_temperature_c: float,
    outdoor_temperature_c: float,
    thermal_output_estimate_kw: float,
    solar_gain_proxy_w_m2: float,
) -> float:
    return (
        model.intercept_c
        + model.room_temperature_coefficient * room_temperature_c
        + model.mass_temperature_coefficient * mass_temperature_c
        + model.outdoor_temperature_coefficient * outdoor_temperature_c
        + model.thermal_output_coefficient * thermal_output_estimate_kw
        + model.solar_gain_coefficient * solar_gain_proxy_w_m2
    )


def _validate_model(
    *,
    model: RoomThermalModel,
    holdout_rows: Sequence[IdentificationDatasetRow],
    holdout_mass: Sequence[float],
    total_valid_rows: int,
    training_rows: int,
) -> RoomThermalValidationReport:
    horizons_minutes = (60, 360, 1440)
    metrics: list[RoomThermalValidationMetric] = []

    for horizon_minutes in horizons_minutes:
        horizon_steps = horizon_minutes // model.interval_minutes
        if horizon_steps <= 0:
            continue

        errors: list[float] = []
        for start_index in range(len(holdout_rows) - horizon_steps):
            current_row = holdout_rows[start_index]
            predicted_room = current_row.room_temperature_c
            if predicted_room is None:
                continue
            predicted_mass = holdout_mass[start_index]

            for offset in range(horizon_steps):
                row = holdout_rows[start_index + offset]
                assert row.outdoor_temperature_c is not None
                assert row.thermal_output_estimate_kw is not None
                assert row.solar_gain_proxy_w_m2 is not None
                predicted_room = _predict_next_room_temperature(
                    model,
                    room_temperature_c=predicted_room,
                    mass_temperature_c=predicted_mass,
                    outdoor_temperature_c=row.outdoor_temperature_c,
                    thermal_output_estimate_kw=row.thermal_output_estimate_kw,
                    solar_gain_proxy_w_m2=row.solar_gain_proxy_w_m2,
                )
                predicted_mass = (
                    model.mass_decay_per_step * predicted_mass
                    + (1.0 - model.mass_decay_per_step) * predicted_room
                )

            actual_room = holdout_rows[start_index + horizon_steps].room_temperature_c
            if actual_room is None:
                continue
            errors.append(predicted_room - actual_room)

        if not errors:
            continue

        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = sqrt(sum(error * error for error in errors) / len(errors))
        bias = sum(errors) / len(errors)
        metrics.append(
            RoomThermalValidationMetric(
                horizon_minutes=horizon_minutes,
                horizon_steps=horizon_steps,
                sample_count=len(errors),
                mae_c=mae,
                rmse_c=rmse,
                bias_c=bias,
            )
        )

    return RoomThermalValidationReport(
        total_valid_rows=total_valid_rows,
        training_rows=training_rows,
        holdout_rows=len(holdout_rows),
        metrics=metrics,
    )
